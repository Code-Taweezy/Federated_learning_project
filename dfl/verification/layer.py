"""Post-acceptance verification layer for decentralised federated learning.

This module implements a four-phase verification mechanism that runs after
each aggregation round. It conservatively flags suspicious neighbours,
rescues falsely rejected ones, re-aggregates when changes are made, and
updates trust scores. The entire layer is designed to be pluggable: the
simulator instantiates either a ``VerificationLayer`` or a
``NoOpVerificationLayer`` depending on the configuration.

Verification hyperparameter rationale:

    verification_epsilon (0.05): The minimum validation-accuracy improvement
        required before flagging or accepting a peer. Chosen based on the
        typical standard deviation of honest node accuracy fluctuations.

    trust_decay (0.95): Exponential moving-average decay factor. With
        beta = 0.95 the effective half-life is about 14 rounds, so
        recent behaviour has roughly 5x the weight of behaviour 30
        rounds ago.

    trust_initial (0.5): Neutral starting point. All nodes begin with
        equal trust scores and must earn higher scores through consistently
        benign behaviour.

    trust_penalty (0.2): A single flag drops trust from 0.5 to 0.3,
        which is below the Phase 1 low-trust threshold of 0.35. This
        ensures that one confirmed flag has immediate protective effect.

    trust_boost (0.05): Conservative recovery rate. Four successful
        rescue rounds are needed to recover from one penalty, making
        it hard for an intermittent attacker to maintain high trust.

    z_low (1.5): Lower bound of the ambiguous z-score range. Messages
        with z-scores below 1.5 sigma (computed against the population of
        all edge distances in the round) are clearly benign and ignored.

    z_high (2.5): Upper bound. Messages above 2.5 sigma are treated as
        clearly anomalous by the round-level anomaly detector, so the
        verification layer focuses only on the ambiguous middle band.

    _PHASE2_MIN_ACCURACY (0.05): Do not attempt rescue when the current
        model accuracy is essentially random (below 5%), since any
        performance comparison would be unreliable.

    _FLAG_CONSECUTIVE_REQUIRED (2): Default for consecutive suspicious
        rounds before acting. Can be overridden via config
        (phase1_consecutive_required).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts, model_distance
from dfl.verification.trust import TrustManager


@dataclass
class RoundContext:
    #All the data the verification layer needs from a single round.

    round_num: int
    current_states: List[Dict[str, torch.Tensor]]
    pre_agg_states: List[Dict[str, torch.Tensor]]
    drifts: List[float]
    peer_devs: List[float]
    z_scores: List[float]
    accepted_per_node: List[List[int]]
    rejected_per_node: List[List[int]]
    neighbor_models: Dict[Tuple[int, int], Dict[str, torch.Tensor]]


class VerificationLayer:
    """Four-phase post-acceptance verification.

    Phase 1: flag (conservatively remove) already-accepted neighbours
        whose messages are persistently suspicious.
    Phase 2: rescue previously rejected neighbours whose messages look
        benign and whose inclusion does not hurt validation.
    Phase 3: re-aggregate the model if Phase 1 or Phase 2 made changes.
    Phase 4: conservative trust-score decay for nodes not touched by
        Phase 1 or Phase 2.
    """

    # Minimum baseline accuracy before Phase 2 rescue is attempted
    _PHASE2_MIN_ACCURACY = 0.05

    # Number of consecutive suspicious rounds required before flagging
    _FLAG_CONSECUTIVE_REQUIRED = 2

    def __init__(self, config: SimulationConfig, nodes, graph):
        self.config = config
        self.nodes = nodes
        self.graph = graph
        self.trust = TrustManager(config)

        # Use config values for thresholds (fall back to class defaults)
        self._phase2_min_accuracy = getattr(config, "phase2_min_accuracy", self._PHASE2_MIN_ACCURACY)
        self._flag_consecutive_required = getattr(config, "phase1_consecutive_required", self._FLAG_CONSECUTIVE_REQUIRED)
        self._phase1_trust_threshold = getattr(config, "phase1_trust_threshold", 0.35)
        self._phase1_min_signals = getattr(config, "phase1_min_signals", 3)
        self._phase2_trust_threshold = getattr(config, "phase2_trust_threshold", 0.4)
        self._phase2_min_signals = getattr(config, "phase2_min_signals", 4)
        self._phase2_drift_sigma_factor = getattr(config, "phase2_drift_sigma_factor", 1.0)

        # Per-round verification output
        self.flags: List[dict] = []
        self.flags_per_round: List[List[dict]] = []
        self.time_per_round: List[float] = []

        # Warmup: do not run verification until enough history has accumulated
        self.warmup_rounds = max(
            3, min(config.num_rounds, config.verification_history_window)
        )

    # Public API

    def run_verification(self, ctx: RoundContext) -> bool:
        """Execute all four verification phases for one round.

        Returns True if any model was modified.
        """
        # Compute population-level distance statistics across all edges
        self._global_dist_mean, self._global_dist_std = self._build_global_distance_stats(
            ctx.pre_agg_states, ctx.neighbor_models
        )

        # Update history before running the phases
        self.trust.update_history(ctx.drifts, ctx.peer_devs)

        self.flags = []
        any_changed = False
        original_states = list(ctx.current_states)
        pending_updates: Dict[int, Dict[str, torch.Tensor]] = {}

        for node_idx in range(self.config.num_nodes):
            S = list(ctx.accepted_per_node[node_idx])
            R = list(ctx.rejected_per_node[node_idx])

            if not R and not S:
                continue

            theta_agg = original_states[node_idx]
            acc_agg = self._evaluate_model(node_idx, theta_agg)
            pre_agg_state = ctx.pre_agg_states[node_idx]

            changed = False

            # Phase 1: flag suspicious accepted neighbours
            if self._phase1(
                node_idx, S, R, theta_agg, acc_agg,
                ctx.z_scores, original_states, ctx.round_num, pre_agg_state,
            ):
                changed = True

            # Phase 2: rescue benign rejected neighbours
            if self._phase2(
                node_idx, S, R, theta_agg, acc_agg,
                ctx.z_scores, ctx.drifts, ctx.peer_devs,
                original_states, ctx.round_num, pre_agg_state,
            ):
                changed = True

            # Phase 3: re-aggregate if anything changed
            if changed:
                accepted_models = [
                    self._get_neighbor_model(node_idx, j, original_states, ctx.neighbor_models)
                    for j in S
                ]
                if accepted_models:
                    theta_final = self._re_aggregate(node_idx, accepted_models, pre_agg_state)
                    pending_updates[node_idx] = theta_final
                any_changed = True

            ctx.accepted_per_node[node_idx] = S
            ctx.rejected_per_node[node_idx] = R

        # Apply all pending model updates at once
        for idx, state in pending_updates.items():
            self.nodes[idx].set_model_state(state)
            ctx.current_states[idx] = state

        # Phase 4: conservative trust decay
        modified_nodes = {
            f["node_id"] for f in self.flags
            if f.get("action") in ("flagged", "rescued")
        }
        self.trust.decay_update(ctx.z_scores, modified_nodes)

        return any_changed

    # Phase implementations

    def _phase1(self, node_idx, S, R, theta_agg, acc_agg,
                z_scores, original_states, round_num, pre_agg_state):
        """Phase 1: conservatively remove accepted neighbours whose messages
        look persistently suspicious and whose removal measurably improves
        validation accuracy.
        """
        if round_num < self.warmup_rounds:
            return False

        eps = self.config.verification_epsilon
        msg_stats = self._build_message_stats(node_idx, S, R, pre_agg_state, original_states)
        if not msg_stats:
            return False

        original_S = list(S)
        nodes_to_flag = []

        hist_ready = len([h for h in self.trust.drift_history if h]) >= 1
        pop_mean_drift, pop_mean_peer = self.trust.get_population_means()

        for j in list(S):
            message_info = msg_stats.get(j)
            if not message_info:
                continue

            msg_z = abs(message_info["z_score"])
            if msg_z < self.config.z_low:
                self.trust.suspicion_counts[(node_idx, j)] = 0
                continue

            s_without_j = [k for k in original_S if k != j]
            if not s_without_j:
                self.trust.suspicion_counts[(node_idx, j)] = 0
                continue

            models_without_j = [
                self._get_neighbor_model(node_idx, k, original_states, {})
                for k in s_without_j
            ]
            theta_without_j = self._re_aggregate(node_idx, models_without_j, pre_agg_state)
            acc_without_j = self._evaluate_model(node_idx, theta_without_j)
            delta_perf = acc_without_j - acc_agg

            sig_high_drift = bool(
                self.trust.drift_history[j]
                and (sum(self.trust.drift_history[j]) / len(self.trust.drift_history[j]))
                > pop_mean_drift
            )
            sig_high_pdev = bool(
                self.trust.peer_dev_history[j]
                and (sum(self.trust.peer_dev_history[j]) / len(self.trust.peer_dev_history[j]))
                > pop_mean_peer
            )
            sig_message_far = message_info["peer_distance"] >= message_info["distance_to_own"]

            signals = sum([sig_high_drift, sig_high_pdev, sig_message_far])

            # Require sustained suspicious behaviour across multiple signals
            suspicious = delta_perf > eps and signals >= self._phase1_min_signals and hist_ready
            suspicion_key = (node_idx, j)
            if suspicious:
                self.trust.suspicion_counts[suspicion_key] = (
                    self.trust.suspicion_counts.get(suspicion_key, 0) + 1
                )
            else:
                self.trust.suspicion_counts[suspicion_key] = 0

            if self.trust.suspicion_counts[suspicion_key] < self._flag_consecutive_required:
                continue

            nodes_to_flag.append(
                (j, delta_perf, message_info,
                 sig_high_drift, sig_high_pdev, sig_message_far)
            )

        changed = False
        for (j, delta_perf, message_info,
             sig_high_drift, sig_high_pdev, sig_message_far) in nodes_to_flag:
            if j not in S:
                continue
            trust_before = self.trust.trust_scores[j]
            S.remove(j)
            if j not in R:
                R.append(j)
            trust_after = self.trust.penalize(j)
            self.trust.suspicion_counts[(node_idx, j)] = 0
            changed = True

            self.flags.append({
                "node_id": j,
                "target_node": node_idx,
                "phase": 1,
                "action": "flagged",
                "delta_perf": float(delta_perf),
                "trust_before": float(trust_before),
                "trust_after": float(trust_after),
                "message_stats": message_info,
                "signals": {
                    "high_drift": sig_high_drift,
                    "high_peer_dev": sig_high_pdev,
                    "message_far_from_peers": sig_message_far,
                },
            })

        return changed

    def _phase2(self, node_idx, S, R, theta_agg, acc_agg,
                z_scores, drifts, peer_devs, original_states, round_num, pre_agg_state):
        """Phase 2: rescue previously rejected neighbours when their message
        looks benign and adding them back does not hurt validation accuracy.
        """
        if acc_agg < self._phase2_min_accuracy:
            return False

        eps = self.config.verification_epsilon
        msg_stats = self._build_message_stats(node_idx, S, R, pre_agg_state, original_states)
        if not msg_stats:
            return False

        sorted_pdevs = sorted(peer_devs)
        median_peer_dev = (
            float(sorted_pdevs[len(sorted_pdevs) // 2]) if sorted_pdevs else 0.0
        )
        mu_drift = float(np.mean(drifts)) if drifts else 0.0
        sigma_drift = float(np.std(drifts)) if drifts else 0.0

        changed = False
        for k in list(R):
            message_info = msg_stats.get(k)
            if not message_info:
                continue

            sig_trust_ok = self.trust.trust_scores[k] >= self._phase2_trust_threshold
            sig_pdev_ok = peer_devs[k] <= median_peer_dev if peer_devs else True
            sig_drift_ok = drifts[k] <= mu_drift + self._phase2_drift_sigma_factor * sigma_drift if drifts else True
            sig_message_ok = abs(message_info["z_score"]) < self.config.z_high

            s_with_k = S + [k]
            models_with_k = [
                self._get_neighbor_model(node_idx, j, original_states, {})
                for j in s_with_k
            ]
            theta_with_k = self._re_aggregate(node_idx, models_with_k, pre_agg_state)
            acc_with_k = self._evaluate_model(node_idx, theta_with_k)
            sig_perf_ok = acc_with_k >= acc_agg - eps

            signals = sum([sig_trust_ok, sig_pdev_ok, sig_drift_ok, sig_message_ok, sig_perf_ok])
            if signals >= self._phase2_min_signals and sig_perf_ok:
                R.remove(k)
                if k not in S:
                    S.append(k)
                old_trust = self.trust.trust_scores[k]
                new_trust = self.trust.boost(k)
                self.trust.suspicion_counts[(node_idx, k)] = 0
                changed = True
                self.flags.append({
                    "node_id": k,
                    "target_node": node_idx,
                    "phase": 2,
                    "action": "rescued",
                    "delta_perf": float(acc_with_k - acc_agg),
                    "trust_before": float(old_trust),
                    "trust_after": float(new_trust),
                    "message_stats": message_info,
                    "signals": {
                        "trust_ok": sig_trust_ok,
                        "peer_dev_ok": sig_pdev_ok,
                        "drift_ok": sig_drift_ok,
                        "message_ok": sig_message_ok,
                        "performance_ok": sig_perf_ok,
                    },
                })
        return changed

    # Helper methods

    def _evaluate_model(self, node_idx, model_state):
        """Evaluate a model state dict on a node's local test set without
        modifying the node's actual model.
        """
        node = self.nodes[node_idx]
        original_state = node.get_model_state()
        node.set_model_state(model_state)
        acc, _ = node.evaluate()
        node.set_model_state(original_state)
        return acc

    def _re_aggregate(self, node_idx, accepted_models, pre_agg_state=None):
        """Re-aggregate using the blending formula: alpha * own + (1-alpha) * mean(accepted)."""
        if not accepted_models:
            return self.nodes[node_idx].get_model_state()

        alpha = self.config.balance_alpha
        own_model = (
            pre_agg_state
            if pre_agg_state is not None
            else self.nodes[node_idx].get_model_state()
        )

        avg = average_state_dicts(accepted_models)
        result = {}
        for key in own_model.keys():
            result[key] = alpha * own_model[key] + (1 - alpha) * avg[key]
        return result

    def _get_neighbor_model(self, node_idx, neighbor_idx, fallback_states, neighbor_models_map):
        """Return the raw model that neighbor_idx sent to node_idx this round."""
        key = (node_idx, neighbor_idx)
        if key in neighbor_models_map:
            return neighbor_models_map[key]
        return fallback_states[neighbor_idx]

    def _message_z_scores(self, values: List[float]) -> List[float]:
        """Return z-scores for a list of scalar message features."""
        if not values:
            return []
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        if std_v <= 1e-12:
            return [0.0 for _ in values]
        return [float((v - mean_v) / std_v) for v in values]

    def _build_global_distance_stats(self, pre_agg_states, neighbor_models):
        """Compute mean and std of L2 distances across all edges in the round.

        This provides a population-level baseline so that per-message z-scores
        are meaningful even when a node has very few neighbours (e.g. ring = 2).
        """
        all_distances = []
        for (receiver, sender), model in neighbor_models.items():
            d = model_distance(pre_agg_states[receiver], model)
            all_distances.append(d)

        if len(all_distances) < 2:
            return 0.0, 1.0

        mean_d = float(np.mean(all_distances))
        std_d = float(np.std(all_distances))
        if std_d < 1e-12:
            std_d = 1.0
        return mean_d, std_d

    def _build_message_stats(self, node_idx, accepted_ids, rejected_ids,
                             pre_agg_state, fallback_states):
        """Build message-level verification features for every neighbour."""
        neighbor_ids = list(accepted_ids) + list(rejected_ids)
        if not neighbor_ids:
            return {}

        neighbor_models_map = getattr(self, '_current_neighbor_models', {})

        messages = {
            nid: self._get_neighbor_model(node_idx, nid, fallback_states, neighbor_models_map)
            for nid in neighbor_ids
        }
        distances_to_own = {
            nid: model_distance(pre_agg_state, msg_state)
            for nid, msg_state in messages.items()
        }
        # Use population-level stats for z-scores instead of local stats
        global_mean = self._global_dist_mean
        global_std = self._global_dist_std
        z_by_neighbor = {
            nid: float((dist - global_mean) / global_std)
            for nid, dist in distances_to_own.items()
        }

        stats = {}
        for nid in neighbor_ids:
            other_models = [messages[oid] for oid in neighbor_ids if oid != nid]
            if other_models:
                median_like = average_state_dicts(other_models)
                peer_distance = model_distance(messages[nid], median_like)
            else:
                peer_distance = 0.0

            stats[nid] = {
                "distance_to_own": float(distances_to_own[nid]),
                "peer_distance": float(peer_distance),
                "z_score": float(z_by_neighbor.get(nid, 0.0)),
            }
        return stats


class NoOpVerificationLayer:
    """A verification layer that does nothing.

    Used when verification is disabled or when the aggregation is FedAvg
    (which accepts everything and has nothing to verify).
    """

    def __init__(self, config=None, nodes=None, graph=None):
        self.trust = type("FakeTrust", (), {"trust_scores": [0.5] * (config.num_nodes if config else 0)})()
        self.flags = []
        self.flags_per_round = []
        self.time_per_round = []

    def run_verification(self, ctx) -> bool:
        return False
