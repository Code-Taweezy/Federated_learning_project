"""Decentralised federated learning simulator.

This is the main file that ties together the network topology,
nodes, attacks, aggregation algorithms, verification layer, and metrics
tracking. It coordinates the training loop and produces results.
"""

import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dfl.config import SimulationConfig
from dfl.network import NetworkGraph
from dfl.attacks import create_attacker
from dfl.metrics import MetricsTracker
from dfl.node import FederatedNode
from dfl.utils import set_global_seed, get_device
from dfl.verification.layer import VerificationLayer, NoOpVerificationLayer, RoundContext


class DecentralisedSimulator:
    """Main simulator for decentralised federated learning."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        set_global_seed(config.seed)
        self.device = get_device()

        # Load dataset
        self._load_data()

        # Create network graph
        self.graph = NetworkGraph(config.num_nodes, config.topology, config.k_neighbors)

        # Initialize nodes
        self._initialize_nodes()

        # Initialize attacker
        self.attacker = create_attacker(config)

        # Initialize metrics tracker
        compromised = self.attacker.compromised_nodes if self.attacker else set()
        self.metrics = MetricsTracker(config.num_nodes, compromised)

        # Results tracking
        self.results = {
            "accuracies": [],
            "losses": [],
            "acceptance_rates": [],
            "honest_accuracies": [],
            "compromised_accuracies": [],
            "drift_per_round": [],
            "peer_deviation_per_round": [],
            "consensus_score_per_round": [],
            "regression_slope_per_round": [],
            "detection_flags_per_round": [],
            "detection_metrics_per_round": [],
            "overhead_time": {"with_detection": [], "without_detection": []},
            "communication_overhead": [],
        }

        self._previous_states = None

        # Initialize verification layer
        if config.verification_enabled:
            self.verification_layer = VerificationLayer(config, self.nodes, self.graph)
        else:
            self.verification_layer = NoOpVerificationLayer(config)

        # Per-round tracking for verification
        self._verification_flags_per_round: List[List[dict]] = []
        self._verification_time_per_round: List[float] = []
        self._round_accepted: List[List[int]] = []
        self._round_rejected: List[List[int]] = []
        self._round_neighbor_models: Dict[Tuple[int, int], Dict] = {}

    def _load_data(self):
        """Load and partition the dataset."""
        from leaf_datasets import load_leaf_dataset, create_leaf_client_partitions

        data_path = f"./leaf/data/{self.config.dataset}/data"
        train_ds, test_ds, model_template, num_classes, input_size = load_leaf_dataset(
            self.config.dataset, data_path
        )

        self.train_dataset = train_ds
        self.test_dataset = test_ds
        self.model_template = model_template
        self.num_classes = num_classes

        train_partitions, test_partitions = create_leaf_client_partitions(
            train_ds, test_ds, self.config.num_nodes, self.config.seed,
            alpha=self.config.partition_alpha,
        )
        self.train_partitions = train_partitions
        self.test_partitions = test_partitions

        print(f"Loaded {self.config.dataset} dataset")
        print(f"   Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    def _initialize_nodes(self):
        """Create all federated nodes with their models and data loaders."""
        from leaf_datasets import LEAFFEMNISTModel, LEAFCelebAModel, LEAFShakespeareModel

        self.nodes = []
        for i in range(self.config.num_nodes):
            torch.manual_seed(self.config.seed + i)
            if self.config.dataset == "femnist":
                model = LEAFFEMNISTModel(self.num_classes).to(self.device)
            elif self.config.dataset == "celeba":
                model = LEAFCelebAModel(self.num_classes).to(self.device)
            elif self.config.dataset == "shakespeare":
                model = LEAFShakespeareModel(self.num_classes).to(self.device)
            else:
                raise ValueError(f"Unsupported dataset: {self.config.dataset}")

            train_subset = Subset(self.train_dataset, self.train_partitions[i])
            test_subset = Subset(self.test_dataset, self.test_partitions[i])

            train_loader = DataLoader(
                train_subset, batch_size=self.config.batch_size,
                shuffle=True, num_workers=0,
            )
            test_loader = DataLoader(
                test_subset, batch_size=512,
                shuffle=False, num_workers=0,
            )

            node = FederatedNode(
                i, model, train_loader, test_loader,
                self.config, self.device, self.config.num_rounds,
            )
            self.nodes.append(node)

        print(f"Initialized {self.config.num_nodes} nodes")
        print(f"   Topology: {self.config.topology}")
        print(f"   Aggregation: {self.config.aggregation}")

    # Per-round metric computations

    def _compute_drift(self, current_states, previous_states):
        """L2 norm of parameter change per node between consecutive rounds."""
        drifts = []
        for curr, prev in zip(current_states, previous_states):
            diff_norm_sq = 0.0
            for key in curr.keys():
                diff = curr[key].float() - prev[key].float()
                diff_norm_sq += torch.sum(diff * diff).item()
            val = float(np.sqrt(diff_norm_sq))
            drifts.append(val if np.isfinite(val) else 0.0)
        return drifts

    def _compute_peer_deviation(self, all_states):
        """Mean L2 distance of each node to its neighbours plus global consensus."""
        deviations = []
        for node_idx in range(self.config.num_nodes):
            neighbors = self.graph.get_neighbors(node_idx)
            own = all_states[node_idx]
            dists = []
            for n_idx in neighbors:
                d = 0.0
                for key in own.keys():
                    diff = own[key].float() - all_states[n_idx][key].float()
                    d += torch.sum(diff * diff).item()
                val = float(np.sqrt(d))
                dists.append(val if np.isfinite(val) else 0.0)
            deviations.append(float(np.mean(dists)) if dists else 0.0)
        mean_dev = float(np.mean(deviations))
        consensus = 1.0 / (1.0 + mean_dev)
        return deviations, consensus

    def _compute_regression_slope(self, values, window=10):
        """OLS slope and R-squared over the last window entries."""
        if len(values) < 2:
            return 0.0, 0.0
        w = values[-window:]
        n = len(w)
        x = np.arange(n, dtype=float)
        y = np.array(w, dtype=float)
        x_m, y_m = np.mean(x), np.mean(y)
        ss_xy = float(np.sum((x - x_m) * (y - y_m)))
        ss_xx = float(np.sum((x - x_m) ** 2))
        ss_yy = float(np.sum((y - y_m) ** 2))
        if ss_xx == 0:
            return 0.0, 0.0
        slope = ss_xy / ss_xx
        r_sq = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0
        return float(slope), float(r_sq)

    def _detect_anomalies(self, drifts, round_num):
        """Z-score based anomaly detection; returns per-node flag dicts."""
        if not drifts or len(drifts) < 2:
            return []
        mean_d = float(np.mean(drifts))
        std_d = float(np.std(drifts))
        flags = []
        for node_id, drift in enumerate(drifts):
            z = (drift - mean_d) / std_d if std_d > 1e-12 else 0.0
            anomaly_score = float(abs(z))
            flagged = anomaly_score > 2.0
            flags.append({
                "node_id": node_id,
                "drift": float(drift),
                "z_score": float(z),
                "anomaly_score": anomaly_score,
                "flagged": flagged,
            })
        return flags

    def _compute_communication_overhead(self) -> int:
        """Estimate total bytes transmitted this round.

        Each node sends its full model to every neighbour.  The cost is
        the number of edges times the model size in bytes (float32).
        """
        # Model size: count total parameters * 4 bytes (float32)
        sample_state = self.nodes[0].get_model_state()
        num_params = sum(p.numel() for p in sample_state.values())
        bytes_per_model = num_params * 4

        total_edges = sum(
            len(self.graph.get_neighbors(i))
            for i in range(self.config.num_nodes)
        )
        return total_edges * bytes_per_model

    def _print_results_table_header(self):
        """Print the column header for round metrics output."""
        if self.attacker:
            header = (
                f"{'Round':>5} | {'Avg Acc':>8} | {'Std Acc':>8} | {'Avg Loss':>8} | "
                f"{'Honest':>8} | {'Compromised':>11}"
            )
        else:
            header = f"{'Round':>5} | {'Avg Acc':>8} | {'Std Acc':>8} | {'Avg Loss':>8}"
        print("\nRound Metrics")
        print(header)
        print("-" * len(header))

    # Main simulation loop

    def run(self):
        """Run the full simulation and return results."""
        print("\nStarting simulation")
        self._print_results_table_header()

        # Initial evaluation
        self._evaluate_round(0)

        # Store initial states for drift computation
        self._previous_states = [node.get_model_state() for node in self.nodes]

        # Round-0 baseline metrics
        peer_devs_0, consensus_0 = self._compute_peer_deviation(self._previous_states)
        self.results["drift_per_round"].append(
            {"mean": 0.0, "std": 0.0, "per_node": [0.0] * self.config.num_nodes}
        )
        self.results["peer_deviation_per_round"].append(
            {"mean": float(np.mean(peer_devs_0)), "per_node": peer_devs_0}
        )
        self.results["consensus_score_per_round"].append(consensus_0)
        avg_accs_0 = [float(np.mean(a)) for a in self.results["accuracies"]]
        slope_0, r_sq_0 = self._compute_regression_slope(avg_accs_0, window=10)
        self.results["regression_slope_per_round"].append(
            {"slope": slope_0, "r_squared": r_sq_0}
        )
        self.results["detection_flags_per_round"].append([])
        self.results["detection_metrics_per_round"].append(
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "asr": 0.0}
        )
        self.results["overhead_time"]["without_detection"].append(0.0)
        self.results["overhead_time"]["with_detection"].append(0.0)
        self.results["communication_overhead"].append(0)
        self._verification_flags_per_round.append([])
        self._verification_time_per_round.append(0.0)

        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            # Local training
            for node in self.nodes:
                node.train_local()

            # Aggregation (timed)
            t_agg_start = time.time()
            pre_agg_states = [node.get_model_state() for node in self.nodes]
            self._aggregation_round(round_num)
            t_agg_end = time.time()
            time_without_detection = t_agg_end - t_agg_start

            # Communication overhead: bytes sent this round
            round_bytes = self._compute_communication_overhead()

            # Collect current model states
            current_states = [node.get_model_state() for node in self.nodes]

            # Drift computation
            drifts = self._compute_drift(current_states, self._previous_states)
            mean_drift = float(np.mean(drifts))
            std_drift = float(np.std(drifts))
            self.results["drift_per_round"].append(
                {"mean": mean_drift, "std": std_drift, "per_node": drifts}
            )

            # Peer deviation and consensus
            peer_devs, consensus = self._compute_peer_deviation(current_states)
            self.results["peer_deviation_per_round"].append(
                {"mean": float(np.mean(peer_devs)), "per_node": peer_devs}
            )
            self.results["consensus_score_per_round"].append(consensus)

            # Anomaly detection (timed) - compute z-score flags, metrics updated later
            t_det_start = time.time()
            flags = self._detect_anomalies(drifts, round_num)
            t_det_end = time.time()
            detection_overhead = t_det_end - t_det_start
            self.results["detection_flags_per_round"].append(flags)

            z_scores = (
                [f["z_score"] for f in flags]
                if flags
                else [0.0] * self.config.num_nodes
            )

            # Verification layer (post-acceptance)
            self.verification_layer.flags = []
            t_ver_start = time.time()

            ctx = RoundContext(
                round_num=round_num,
                current_states=current_states,
                pre_agg_states=pre_agg_states,
                drifts=drifts,
                peer_devs=peer_devs,
                z_scores=z_scores,
                accepted_per_node=self._round_accepted,
                rejected_per_node=self._round_rejected,
                neighbor_models=self._round_neighbor_models,
            )
            # Give verification layer access to current round's neighbor models
            self.verification_layer._current_neighbor_models = self._round_neighbor_models
            verification_changed = self.verification_layer.run_verification(ctx)

            t_ver_end = time.time()
            verification_time = t_ver_end - t_ver_start

            verification_flags = list(self.verification_layer.flags)
            self._verification_flags_per_round.append(verification_flags)
            self._verification_time_per_round.append(verification_time)

            # Compute detection metrics (once, after verification runs)
            # Combines z-score flags with verification layer flags
            if self.config.verification_enabled:
                # Get set of nodes flagged by verification layer
                ver_flagged_nodes = {
                    vf["node_id"] for vf in verification_flags
                    if vf.get("action") == "flagged"
                }
                # Create combined flags: z-score flags OR verification layer flags
                combined_flags = []
                for node_id in range(self.config.num_nodes):
                    zscore_flagged = any(
                        f["node_id"] == node_id and f.get("flagged", False)
                        for f in flags
                    )
                    ver_layer_flagged = node_id in ver_flagged_nodes
                    is_flagged = zscore_flagged or ver_layer_flagged

                    combined_flags.append({
                        "node_id": node_id,
                        "flagged": is_flagged,
                        "z_score_flagged": zscore_flagged,
                        "verification_flagged": ver_layer_flagged,
                    })

                # Track acceptance events for post-acceptance metrics
                for event in self.verification_layer.acceptance.acceptance_events:
                    if event.get("round") == round_num:
                        self.metrics.record_acceptance(
                            event["node_id"],
                            event["round"],
                        )

                # Track attack start for post-acceptance metrics
                if (
                    self.attacker
                    and round_num == self.config.attack_start_round
                ):
                    for attacker_id in self.attacker.compromised_nodes:
                        self.metrics.record_attack_start(attacker_id, round_num)

                # Record detections with acceptance status for metrics
                for cf in combined_flags:
                    if cf["flagged"]:
                        was_accepted = self.verification_layer.acceptance.was_ever_accepted(
                            cf["node_id"]
                        )
                        self.metrics.record_detection(
                            cf["node_id"],
                            round_num,
                            was_accepted,
                        )

                round_metrics = self.metrics.update_round(round_num, combined_flags)
            else:
                # No verification, just use z-score flags
                round_metrics = self.metrics.update_round(round_num, flags)

            self.results["detection_metrics_per_round"].append({
                "precision": round_metrics["precision"],
                "recall": round_metrics["recall"],
                "f1": round_metrics["f1"],
                "asr": round_metrics["asr"],
            })

            # If verification changed models, re-collect states
            if verification_changed:
                current_states = [node.get_model_state() for node in self.nodes]

            # Evaluation (after verification)
            self._evaluate_round(round_num)

            # Regression slope
            avg_accs = [float(np.mean(a)) for a in self.results["accuracies"]]
            slope, r_sq = self._compute_regression_slope(avg_accs, window=10)
            self.results["regression_slope_per_round"].append(
                {"slope": slope, "r_squared": r_sq}
            )

            # Overhead tracking
            time_with_all = time_without_detection + detection_overhead + verification_time
            self.results["overhead_time"]["without_detection"].append(time_without_detection)
            self.results["overhead_time"]["with_detection"].append(time_with_all)
            self.results["communication_overhead"].append(round_bytes)

            # Structured metrics line for dashboard parsing
            n_flagged = sum(1 for f in flags if f["flagged"]) if flags else 0
            n_ver_flagged = sum(1 for vf in verification_flags if vf["action"] == "flagged")
            n_ver_rescued = sum(1 for vf in verification_flags if vf["action"] == "rescued")

            # Per-round precision, recall, F1 from MetricsTracker
            rm = round_metrics
            print(
                f"METRICS|{round_num}|{mean_drift:.6f}|{std_drift:.6f}"
                f"|{float(np.mean(peer_devs)):.6f}|{consensus:.6f}"
                f"|{slope:.6f}|{r_sq:.6f}"
                f"|{n_flagged}|{self.metrics.cumulative_tp}|{self.metrics.cumulative_fp}"
                f"|{self.metrics.cumulative_tn}|{self.metrics.cumulative_fn}"
                f"|{time_without_detection:.6f}|{time_with_all:.6f}"
                f"|{n_ver_flagged}|{n_ver_rescued}|{verification_time:.6f}"
                f"|{rm['precision']:.6f}|{rm['recall']:.6f}|{rm['f1']:.6f}|{rm['asr']:.6f}"
            )

            self._previous_states = current_states

        # Final summary
        self._print_summary()

        # Add post-acceptance metrics to results if applicable
        if self.attacker and self.config.attack_start_round > 0:
            self.results["post_acceptance"] = self.metrics.get_post_acceptance_summary()

        return self.results

    def _aggregation_round(self, round_num: int):
        """Perform one round of aggregation across all nodes."""
        # Update attacker's round counter for delayed attack activation
        if self.attacker:
            self.attacker.set_current_round(round_num)

        all_states = [node.get_model_state() for node in self.nodes]

        new_states = []
        self._round_accepted = []
        self._round_rejected = []
        self._round_neighbor_models = {}

        for node_idx, node in enumerate(self.nodes):
            neighbors_idx = self.graph.get_neighbors(node_idx)
            own_state = all_states[node_idx]

            neighbor_states = []
            for neighbor_idx in neighbors_idx:
                if self.attacker and self.attacker.is_compromised(neighbor_idx):
                    honest_neighbors = [
                        i for i in neighbors_idx
                        if not self.attacker.is_compromised(i)
                    ]
                    honest_states = [all_states[i] for i in honest_neighbors]
                    malicious_state = self.attacker.craft_malicious_update(
                        honest_states, neighbor_idx
                    )
                    raw_state = malicious_state if malicious_state else all_states[neighbor_idx]
                else:
                    raw_state = all_states[neighbor_idx]

                self._round_neighbor_models[(node_idx, neighbor_idx)] = raw_state
                neighbor_states.append(raw_state)

            new_state, accepted, rejected = node.aggregator.aggregate(
                own_state, neighbor_states, neighbors_idx
            )
            new_states.append(new_state)
            self._round_accepted.append(accepted)
            self._round_rejected.append(rejected)

        for node, new_state in zip(self.nodes, new_states):
            node.set_model_state(new_state)

    def _evaluate_round(self, round_num: int):
        """Evaluate all nodes and print round results."""
        accuracies = []
        losses = []

        for node in self.nodes:
            acc, loss = node.evaluate()
            accuracies.append(acc)
            losses.append(loss)

        self.results["accuracies"].append(accuracies)
        self.results["losses"].append(losses)

        avg_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        avg_loss = float(np.mean(losses))

        if self.attacker:
            honest_accs = [
                accuracies[i] for i in range(self.config.num_nodes)
                if not self.attacker.is_compromised(i)
            ]
            compromised_accs = [
                accuracies[i] for i in range(self.config.num_nodes)
                if self.attacker.is_compromised(i)
            ]

            honest_mean = float(np.mean(honest_accs)) if honest_accs else 0.0
            compromised_mean = float(np.mean(compromised_accs)) if compromised_accs else 0.0

            self.results["honest_accuracies"].append(honest_mean)
            self.results["compromised_accuracies"].append(compromised_mean)

            print(
                f"{round_num:>5} | {avg_acc:>8.4f} | {std_acc:>8.4f} | {avg_loss:>8.4f} | "
                f"{honest_mean:>8.4f} | {compromised_mean:>11.4f}"
            )
        else:
            print(f"{round_num:>5} | {avg_acc:>8.4f} | {std_acc:>8.4f} | {avg_loss:>8.4f}")

    def _print_summary(self):
        """Print final summary of the simulation."""
        print("\nFinal Results")

        final_accs = self.results["accuracies"][-1]
        print(f"Final Accuracy: {np.mean(final_accs):.4f} +/- {np.std(final_accs):.4f}")

        if self.attacker:
            honest_acc = self.results["honest_accuracies"][-1]
            comp_acc = self.results["compromised_accuracies"][-1]
            print(f"Honest Nodes: {honest_acc:.4f}")
            print(f"Compromised Nodes: {comp_acc:.4f}")
            print(f"Attack Impact: {honest_acc - comp_acc:.4f}")

        # Drift summary
        if self.results["drift_per_round"]:
            last_drift = self.results["drift_per_round"][-1]
            print(
                f"\nDrift (last round):  mean={last_drift['mean']:.6f}  "
                f"std={last_drift['std']:.6f}"
            )

        if self.results["consensus_score_per_round"]:
            print(
                f"Consensus Score:     "
                f"{self.results['consensus_score_per_round'][-1]:.6f}"
            )

        if self.results["regression_slope_per_round"]:
            last_reg = self.results["regression_slope_per_round"][-1]
            print(
                f"Regression Slope:    slope={last_reg['slope']:.6f}  "
                f"R²={last_reg['r_squared']:.6f}"
            )

        # Detection metrics from MetricsTracker
        if self.attacker:
            summary = self.metrics.get_summary()
            print(f"\nDetection Metrics:")
            print(f"  True Positives:   {summary['true_positives']}")
            print(f"  False Positives:  {summary['false_positives']}")
            print(f"  True Negatives:   {summary['true_negatives']}")
            print(f"  False Negatives:  {summary['false_negatives']}")
            print(f"  Precision:        {summary['precision']:.4f}")
            print(f"  Recall:           {summary['recall']:.4f}")
            print(f"  F1 Score:         {summary['f1_score']:.4f}")
            if summary["detection_time"] is not None:
                print(f"  Detection Time:   T_detect = round {summary['detection_time']}")
            else:
                print(f"  Detection Time:   not detected")

        if self.results["overhead_time"]["with_detection"]:
            avg_wo = float(np.mean(self.results["overhead_time"]["without_detection"]))
            avg_w = float(np.mean(self.results["overhead_time"]["with_detection"]))
            print(f"\nOverhead (avg per round):")
            print(f"  Without detection: {avg_wo:.4f}s")
            print(f"  With detection:    {avg_w:.4f}s")

        if self.results["communication_overhead"]:
            total_bytes = sum(self.results["communication_overhead"])
            avg_bytes = total_bytes / max(1, len(self.results["communication_overhead"]))
            print(f"\nCommunication:")
            print(f"  Total:   {total_bytes / 1e6:.2f} MB")
            print(f"  Per round: {avg_bytes / 1e6:.2f} MB")

        # Verification layer summary
        if self.config.verification_enabled:
            total_p1 = sum(
                sum(1 for vf in rflags if vf["action"] == "flagged")
                for rflags in self._verification_flags_per_round
            )
            total_p2 = sum(
                sum(1 for vf in rflags if vf["action"] == "rescued")
                for rflags in self._verification_flags_per_round
            )
            print(f"\nVerification Layer:")
            print(f"  Phase 1 flags (total): {total_p1}")
            print(f"  Phase 2 rescues (total): {total_p2}")
            print(
                f"  Final trust scores: "
                f"{[round(t, 3) for t in self.verification_layer.trust.trust_scores]}"
            )

    def save_results(self, filepath: str):
        """Save results to JSON with run timestamp and per-round table."""
        from datetime import datetime as _dt

        now = _dt.now()

        round_results = []
        for r in range(len(self.results["accuracies"])):
            accs = self.results["accuracies"][r]
            losses = self.results["losses"][r]
            row = {
                "round": r,
                "avg_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "avg_loss": float(np.mean(losses)),
                "honest_accuracy": (
                    float(self.results["honest_accuracies"][r])
                    if self.attacker and r < len(self.results["honest_accuracies"])
                    else None
                ),
                "compromised_accuracy": (
                    float(self.results["compromised_accuracies"][r])
                    if self.attacker and r < len(self.results["compromised_accuracies"])
                    else None
                ),
            }

            adv_idx = r
            if adv_idx < len(self.results["drift_per_round"]):
                dr = self.results["drift_per_round"][adv_idx]
                row["drift_mean"] = dr["mean"]
                row["drift_std"] = dr["std"]
            if adv_idx < len(self.results["peer_deviation_per_round"]):
                row["peer_deviation_mean"] = self.results["peer_deviation_per_round"][adv_idx]["mean"]
            if adv_idx < len(self.results["consensus_score_per_round"]):
                row["consensus_score"] = self.results["consensus_score_per_round"][adv_idx]
            if adv_idx < len(self.results["regression_slope_per_round"]):
                reg = self.results["regression_slope_per_round"][adv_idx]
                row["regression_slope"] = reg["slope"]
                row["regression_r_squared"] = reg["r_squared"]
            if adv_idx < len(self.results["detection_flags_per_round"]):
                row["detection_flags"] = self.results["detection_flags_per_round"][adv_idx]
            if adv_idx < len(self.results["overhead_time"]["without_detection"]):
                row["time_without_detection"] = self.results["overhead_time"]["without_detection"][adv_idx]
                row["time_with_detection"] = self.results["overhead_time"]["with_detection"][adv_idx]

            if adv_idx < len(self.results.get("communication_overhead", [])):
                row["communication_bytes"] = self.results["communication_overhead"][adv_idx]

            # Verification layer per-round data
            if adv_idx < len(self._verification_flags_per_round):
                vflags = self._verification_flags_per_round[adv_idx]
                row["verification_flags"] = vflags
                row["verification_time"] = (
                    self._verification_time_per_round[adv_idx]
                    if adv_idx < len(self._verification_time_per_round)
                    else 0.0
                )
                row["nodes_flagged_phase1"] = sum(
                    1 for vf in vflags if vf.get("action") == "flagged"
                )
                row["nodes_rescued_phase2"] = sum(
                    1 for vf in vflags if vf.get("action") == "rescued"
                )

            # Per-round detection metrics
            if adv_idx < len(self.metrics.per_round):
                rm = self.metrics.per_round[adv_idx]
                row["round_precision"] = rm["precision"]
                row["round_recall"] = rm["recall"]
                row["round_f1"] = rm["f1"]
                row["round_asr"] = rm["asr"]

            round_results.append(row)

        final_accs = self.results["accuracies"][-1]
        detection_summary = self.metrics.get_summary() if self.attacker else None

        summary = {
            "final_accuracy": float(np.mean(final_accs)),
            "final_accuracy_std": float(np.std(final_accs)),
            "honest_accuracy": (
                float(self.results["honest_accuracies"][-1])
                if self.attacker else None
            ),
            "compromised_accuracy": (
                float(self.results["compromised_accuracies"][-1])
                if self.attacker else None
            ),
            "attack_impact": (
                float(
                    self.results["honest_accuracies"][-1]
                    - self.results["compromised_accuracies"][-1]
                )
                if self.attacker else None
            ),
            "detection": detection_summary,
            "overhead_avg": {
                "without_detection": (
                    float(np.mean(self.results["overhead_time"]["without_detection"]))
                    if self.results["overhead_time"]["without_detection"] else None
                ),
                "with_detection": (
                    float(np.mean(self.results["overhead_time"]["with_detection"]))
                    if self.results["overhead_time"]["with_detection"] else None
                ),
            },
            "communication": {
                "total_bytes": (
                    sum(self.results["communication_overhead"])
                    if self.results.get("communication_overhead") else 0
                ),
                "avg_bytes_per_round": (
                    float(np.mean(self.results["communication_overhead"]))
                    if self.results.get("communication_overhead") else 0
                ),
            },
            "verification": {
                "total_phase1_flags": sum(
                    sum(1 for vf in rflags if vf.get("action") == "flagged")
                    for rflags in self._verification_flags_per_round
                ),
                "total_phase2_rescues": sum(
                    sum(1 for vf in rflags if vf.get("action") == "rescued")
                    for rflags in self._verification_flags_per_round
                ),
                "final_trust_scores": [
                    round(t, 6)
                    for t in self.verification_layer.trust.trust_scores
                ],
            }
            if self.config.verification_enabled and self.config.aggregation != "fedavg"
            else None,
        }

        output = {
            "run_timestamp": now.isoformat(timespec="seconds"),
            "run_date": now.strftime("%Y-%m-%d"),
            "config": {
                "dataset": self.config.dataset,
                "num_nodes": self.config.num_nodes,
                "num_rounds": self.config.num_rounds,
                "topology": self.config.topology,
                "aggregation": self.config.aggregation,
                "attack_ratio": self.config.attack_ratio,
                "attack_type": self.config.attack_type,
            },
            "round_results": round_results,
            "summary": summary,
            "results": self.results,
            "compromised_nodes": (
                list(self.attacker.compromised_nodes) if self.attacker else []
            ),
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filepath}")

        # Export to Google Sheets if configured
        try:
            from sheets_exporter import export_results as _sheets_export
            _sheets_export(filepath)
        except ImportError:
            pass  # sheets_exporter not available
        except Exception as e:
            print(f"[Sheets] Export failed: {e}")
