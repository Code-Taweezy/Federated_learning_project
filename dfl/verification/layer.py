"""Verification layer for decentralized federated learning.

This module implements a post-aggregation verification layer that:
1. Detects attackers that passed the aggregator (false negatives)
2. Rescues honest nodes that were incorrectly rejected (false positives)
3. Re-aggregates with corrected neighbor sets
4. Updates trust scores based on outcomes
5. Tracks acceptance status for post-acceptance attack detection

Why we Analyze what nodes SEND.
- Attackers SEND malicious updates but have normal FINAL states (they receive honest updates)
- Honest nodes SEND normal updates but may have damaged FINAL states (they receive malicious updates)
- So we detect based on SENT updates via neighbor_models, not current_states

Post-Acceptance Attack Detection:
- Tracks when nodes achieve "accepted" status through honest behavior
- Detects behavioral changes when accepted nodes turn malicious
- Uses heightened sensitivity for accepted nodes showing anomalies
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts
from dfl.verification.acceptance import AcceptanceStatus, AcceptanceTracker
from dfl.verification.trust import TrustManager


@dataclass
class RoundContext:
    """All data needed for verification in a single round.

    This dataclass bundles the inputs for run_verification(). Note that
    current_states, accepted_per_node, and rejected_per_node are MUTATED
    by the verification layer when it makes corrections.
    """

    round_num: int
    current_states: List[Dict[str, torch.Tensor]]  # Post-aggregation states (mutable)
    pre_agg_states: List[Dict[str, torch.Tensor]]  # Pre-aggregation states
    drifts: List[float]  # Per-node drift values
    peer_devs: List[float]  # Per-node peer deviation values
    z_scores: List[float]  # Per-node z-scores from anomaly detection
    accepted_per_node: List[List[int]]  # Accepted neighbor IDs per node (mutable)
    rejected_per_node: List[List[int]]  # Rejected neighbor IDs per node (mutable)
    neighbor_models: Dict[Tuple[int, int], Dict[str, torch.Tensor]]  # (receiver, sender) -> model


@dataclass
class SenderStats:
    """Statistics about what senders sent to a particular receiver.

    Used for outlier detection based on robust statistics.
    """

    robust_center: Dict[str, torch.Tensor]  # Coordinate-wise median of sent models
    distances: Dict[int, float]  # sender_id -> distance to robust center
    median_dist: float  # Median of all distances
    mad: float  # Median Absolute Deviation
    outlier_threshold: float  # Distance above which sender is an outlier


class VerificationLayer:
    """Post-aggregation verification to catch missed attackers and rescue honest nodes.

    The verification layer runs after each aggregation round and performs:
    - Phase 1: Detect attackers that passed the aggregator
    - Phase 1.2: Detect post-acceptance attacks (behavioral changes)
    - Phase 2: Rescue honest nodes that were incorrectly rejected
    - Phase 3: Re-aggregate with corrected neighbor sets
    - Phase 4: Update trust scores
    - Phase 5: Update acceptance status

    All detection is based on what nodes SEND (neighbor_models),
    not their final state after aggregation (current_states).

    Post-acceptance detection: An anomaly from an accepted node is more significant
    because it represents a behavioral CHANGE, not baseline behavior.
    """

    def __init__(self, config: SimulationConfig, nodes, graph):
        """Initialize the verification layer.

        Args:
            config: Simulation configuration
            nodes: List of FederatedNode instances
            graph: NetworkGraph for topology information
        """
        self.config = config
        self.nodes = nodes
        self.graph = graph
        self.trust = TrustManager(config)

        # Acceptance tracking for post-acceptance attack detection
        self.acceptance = AcceptanceTracker(
            num_nodes=config.num_nodes,
            acceptance_threshold=config.acceptance_trust_threshold,
            consecutive_clean_required=config.acceptance_consecutive_clean,
            anomaly_revoke_threshold=config.acceptance_anomaly_revoke,
        )

        # Output tracking for metrics
        self.flags: List[dict] = []
        self.flags_per_round: List[List[dict]] = []
        self.time_per_round: List[float] = []

        # Reduce warmup to catch attackers early (they're most distinguishable early)
        self.warmup_rounds = 2

        # Threshold parameters
        self._outlier_k = 1.5  # MAD multiplier for outlier detection (more sensitive)
        self._direction_threshold = 0.40  # Below this = attack (raised to catch more)
        self._debug = True  # Set True to print debug info

        # Post-acceptance sensitivity multiplier
        # Accepted nodes get tighter thresholds (anomaly is more significant)
        self._post_acceptance_sensitivity = 1.5

        # Historical tracking for multi-round evidence (reduces FPs)
        self._flag_history: Dict[int, List[int]] = {}  # sender_id -> list of rounds flagged
        self._history_window = 5  # Look at last N rounds
        self._min_flags_for_action = 2  # Require flags in >=2 of last N rounds

        # Track aggregator rejections separately for validation
        self._aggregator_rejection_history: Dict[int, List[int]] = {}  # sender_id -> rounds

    def run_verification(self, ctx: RoundContext) -> bool:
        """Execute all verification phases for one round.

        Args:
            ctx: RoundContext with all data for this round

        Returns:
            True if any model was modified, False otherwise
        """
        self.flags = []

        # Update trust round counter for trajectory tracking
        self.trust.set_round(ctx.round_num)

        # Skip during warmup period
        if ctx.round_num < self.warmup_rounds:
            self.flags_per_round.append([])
            self.time_per_round.append(0.0)
            return False

        start_time = time.time()

        # Phase 1: Detect attackers that passed the aggregator
        flagged_per_node = self._phase1_detect(ctx)

        # Phase 1b: Detect post-acceptance attacks (behavioral changes)
        post_acceptance_flags = self._detect_post_acceptance_attacks(ctx, flagged_per_node)

        # Merge post-acceptance flags into flagged_per_node
        for node_idx, additional_flags in post_acceptance_flags.items():
            if node_idx not in flagged_per_node:
                flagged_per_node[node_idx] = set()
            flagged_per_node[node_idx].update(additional_flags)

        # Phase 2: Rescue honest nodes that were incorrectly rejected
        rescued_per_node = self._phase2_rescue(ctx, flagged_per_node)

        # Phase 3: Re-aggregate with corrected neighbor sets
        modified_nodes = self._phase3_reaggregate(ctx, flagged_per_node, rescued_per_node)

        # Phase 4: Update trust scores
        self._phase4_update_trust(flagged_per_node, rescued_per_node)

        # Phase 5: Update acceptance status
        all_flagged: Set[int] = set()
        for flags in flagged_per_node.values():
            all_flagged.update(flags)

        status_changes = self.acceptance.update_round(
            ctx.round_num,
            self.trust.trust_scores,
            all_flagged,
        )

        # Log acceptance status changes
        if self._debug and status_changes:
            for node_id, new_status in status_changes.items():
                print(f"[Acceptance] Node {node_id} -> {new_status.value}")

        elapsed = time.time() - start_time
        self.flags_per_round.append(self.flags.copy())
        self.time_per_round.append(elapsed)

        return len(modified_nodes) > 0

    
    # Phase 1: Detect Attackers

    def _phase1_detect(self, ctx: RoundContext) -> Dict[int, Set[int]]:
        """Detect attackers that passed the aggregator (false negatives).

        Uses statistical detection based on BOTH:
        1. Distance from robust center (catches noise attacks)
        2. Anti-correlation with peers (catches directed attacks)

        A sender is flagged if they have sufficient TRUST-WEIGHTED votes from receivers.
        We weight votes by receiver trust to reduce attacker influence on voting.
        Historical tracking requires consistent flags across rounds to reduce FPs.

        Aggregator rejections are VALIDATED against our detection - not blindly trusted.

        Returns:
            Dict mapping node_idx to set of flagged sender IDs
        """
        if self._debug:
            print(f"[Debug] Phase 1 starting for round {ctx.round_num}")

        # Step 0: Track aggregator rejections from TRUSTED receivers only
        # Attackers might reject honest nodes, so we ignore rejections from low-trust nodes
        aggregator_rejected: Set[int] = set()
        for node_idx in range(len(self.nodes)):
            # Only trust rejections from nodes with sufficient trust
            if self.trust.get_trust(node_idx) < 0.35:
                continue

            rejected = ctx.rejected_per_node[node_idx]
            aggregator_rejected.update(rejected)
            # Update rejection history (only from trusted receivers)
            for sender_id in rejected:
                if sender_id not in self._aggregator_rejection_history:
                    self._aggregator_rejection_history[sender_id] = []
                self._aggregator_rejection_history[sender_id].append(ctx.round_num)

        if self._debug and aggregator_rejected:
            print(f"[Debug] Aggregator-rejected senders: {aggregator_rejected}")

        # Step 1: Collect TRUST-WEIGHTED votes from all receivers for accepted senders
        sender_votes: Dict[int, float] = {}  # sender_id -> weighted vote sum
        sender_voter_count: Dict[int, int] = {}  # sender_id -> number of voters

        for node_idx in range(len(self.nodes)):
            accepted = ctx.accepted_per_node[node_idx]

            if len(accepted) < 2:
                continue

            stats = self._compute_sender_stats(node_idx, accepted, ctx.neighbor_models)
            if not stats.distances:
                continue

            receiver_model = ctx.pre_agg_states[node_idx]
            receiver_trust = self.trust.get_trust(node_idx)

            for sender_id in accepted:
                if self._is_sender_outlier(
                    sender_id, stats, node_idx, accepted, ctx.neighbor_models, receiver_model
                ):
                    # Weight vote by receiver's trust score
                    vote_weight = receiver_trust
                    if sender_id not in sender_votes:
                        sender_votes[sender_id] = 0.0
                        sender_voter_count[sender_id] = 0
                    sender_votes[sender_id] += vote_weight
                    sender_voter_count[sender_id] += 1

        if self._debug:
            print(f"[Debug] Weighted votes after Step 1: {sender_votes}")

        # Step 2: Update historical tracking and determine validated flags
        k = self.config.k_neighbors

        # Adaptive threshold based on network connectivity
        weighted_threshold = 0.75  # ~2 receivers at 0.5 trust
        count_threshold = max(2, k // 3)

        # Update flag history for this round
        current_round_flags: Set[int] = set()
        for sender_id, weighted_sum in sender_votes.items():
            voter_count = sender_voter_count[sender_id]
            if weighted_sum >= weighted_threshold and voter_count >= count_threshold:
                current_round_flags.add(sender_id)
                if sender_id not in self._flag_history:
                    self._flag_history[sender_id] = []
                self._flag_history[sender_id].append(ctx.round_num)

        # Step 3: Validate flags based on historical consistency
        validated_flagged: Set[int] = set()

        # 3a: Check our own detection (from sender_votes)
        for sender_id in current_round_flags:
            history = self._flag_history.get(sender_id, [])
            recent_flags = [r for r in history if r >= ctx.round_num - self._history_window]
            strong_signal = sender_votes.get(sender_id, 0) >= 1.5
            consistent = len(recent_flags) >= self._min_flags_for_action
            if consistent or strong_signal:
                validated_flagged.add(sender_id)

        # 3b: Validate aggregator rejections - require stronger evidence to avoid FPs
        for sender_id in aggregator_rejected:
            agg_history = self._aggregator_rejection_history.get(sender_id, [])
            recent_rejections = [r for r in agg_history if r >= ctx.round_num - self._history_window]

            # Also check our flag history
            our_history = self._flag_history.get(sender_id, [])
            recent_our_flags = [r for r in our_history if r >= ctx.round_num - self._history_window]

            # Validate aggregator rejection if:
            # 1. Consistently rejected (>= 3 times in window), OR
            # 2. We also flagged them independently (corroborated), OR
            # 3. Their trust is very low (strong accumulated evidence)
            trust = self.trust.get_trust(sender_id)
            consistent_rejection = len(recent_rejections) >= 3
            corroborated = len(recent_our_flags) >= 1
            very_low_trust = trust < 0.2

            if consistent_rejection or corroborated or very_low_trust:
                validated_flagged.add(sender_id)

        if self._debug:
            print(f"[Debug] Current round flags: {current_round_flags}")
            print(f"[Debug] Validated (with history): {validated_flagged}")

        # Step 4: Distribute validated flags to per-node sets
        flagged_per_node: Dict[int, Set[int]] = {}
        for node_idx in range(len(self.nodes)):
            accepted = ctx.accepted_per_node[node_idx]
            rejected = ctx.rejected_per_node[node_idx]

            # Only include validated flags (both from our detection and aggregator)
            validated_rejected = validated_flagged & set(rejected)
            validated_accepted = validated_flagged & set(accepted)
            node_flagged = validated_rejected | validated_accepted

            if node_flagged:
                flagged_per_node[node_idx] = node_flagged

                # Record flags for metrics
                if accepted:
                    stats = self._compute_sender_stats(node_idx, accepted, ctx.neighbor_models)
                else:
                    stats = SenderStats(
                        robust_center={},
                        distances={},
                        median_dist=0.0,
                        mad=0.0,
                        outlier_threshold=float('inf'),
                    )

                for sender_id in node_flagged:
                    is_aggregator_flag = sender_id in rejected
                    self._record_flag(
                        node_idx=node_idx,
                        sender_id=sender_id,
                        stats=stats,
                        phase=1,
                        action="flagged",
                        aggregator_flagged=is_aggregator_flag,
                    )

        return flagged_per_node

    # Phase 1.2: Post-Acceptance Attack Detection
    

    def _detect_post_acceptance_attacks(
        self,
        ctx: RoundContext,
        initial_flagged: Dict[int, Set[int]],
    ) -> Dict[int, Set[int]]:
        """Detect attacks from previously-accepted nodes.

        Key insight: An anomaly from an accepted node is MORE significant
        because it represents a behavioral CHANGE, not baseline behavior.

        Detection signals:
        1. Trust drop from peak (accepted nodes had high trust)
        2. Trust volatility increase (stable -> unstable)
        3. Negative trust trend (was stable, now declining)

        Combined with direction-based detection using LOWER thresholds
        for accepted nodes (heightened sensitivity).

        Args:
            ctx: Round context with all data
            initial_flagged: Already flagged nodes from Phase 1

        Returns:
            Dict mapping node_idx to set of additional flagged sender IDs
        """
        post_acceptance_flags: Dict[int, Set[int]] = {}

        # Behavioral change thresholds
        trust_drop_threshold = 0.15  # Dropped 15% from peak
        volatility_threshold = 0.08  # High variance in trust
        trend_threshold = -0.02  # Declining trend

        for node_idx in range(len(self.nodes)):
            accepted = ctx.accepted_per_node[node_idx]
            if len(accepted) < 2:
                continue

            for sender_id in accepted:
                # Skip if already flagged by Phase 1
                if sender_id in initial_flagged.get(node_idx, set()):
                    continue

                record = self.acceptance.get_record(sender_id)

                # Only analyze nodes that achieved acceptance at some point
                if record.acceptance_round is None:
                    continue

                # Compute behavioral change signals
                trust_drop = self.trust.compute_trust_drop(sender_id)
                volatility = self.trust.compute_trust_volatility(sender_id)
                trend = self.trust.compute_trust_trend(sender_id)

                # Post-acceptance attack signature:
                # - Significant trust drop from peak (they were trusted!)
                # - Negative trend (trust declining)
                # - High volatility (erratic behavior)
                is_behavioral_change = (
                    (trust_drop >= trust_drop_threshold and trend < 0)
                    or (volatility >= volatility_threshold and trend < trend_threshold)
                )

                if not is_behavioral_change:
                    continue

                # Verify with sender statistics using LOWER thresholds
                stats = self._compute_sender_stats(
                    node_idx, accepted, ctx.neighbor_models
                )

                receiver_model = ctx.pre_agg_states[node_idx]
                direction_score, rel_magnitude = self._compute_direction_score(
                    sender_id, node_idx, accepted, ctx.neighbor_models, receiver_model
                )

                # Lower threshold for post-acceptance detection
                # (multiply by 1.5 = more lenient baseline, so division = tighter)
                effective_threshold = self._direction_threshold * 1.5

                if direction_score < effective_threshold:
                    if node_idx not in post_acceptance_flags:
                        post_acceptance_flags[node_idx] = set()
                    post_acceptance_flags[node_idx].add(sender_id)

                    # Record detailed flag for metrics
                    self._record_flag(
                        node_idx=node_idx,
                        sender_id=sender_id,
                        stats=stats,
                        phase=1,
                        action="flagged",
                        aggregator_flagged=False,
                        post_acceptance=True,
                        behavioral_signals={
                            "trust_drop": trust_drop,
                            "volatility": volatility,
                            "trend": trend,
                            "direction_score": direction_score,
                            "rounds_since_acceptance": ctx.round_num
                            - record.acceptance_round,
                        },
                    )

                    if self._debug:
                        print(
                            f"[Post-Acceptance] Node {sender_id} flagged: "
                            f"drop={trust_drop:.3f}, vol={volatility:.3f}, "
                            f"trend={trend:.3f}, dir_score={direction_score:.3f}"
                        )

        return post_acceptance_flags


    # Phase 2: Rescue Honest Nodes
    

    def _phase2_rescue(
        self, ctx: RoundContext, flagged_per_node: Dict[int, Set[int]]
    ) -> Dict[int, Set[int]]:
        """Rescue honest nodes that were incorrectly rejected (false positives).

        For each node, look at its REJECTED neighbors. Rescue those whose
        SENT updates are consistent with the honest consensus (non-outliers).

        Returns:
            Dict mapping node_idx to set of rescued sender IDs
        """
        rescued_per_node: Dict[int, Set[int]] = {}

        for node_idx in range(len(self.nodes)):
            rejected = ctx.rejected_per_node[node_idx]

            if not rejected:
                continue

            # Build clean reference set (accepted minus flagged)
            accepted = ctx.accepted_per_node[node_idx]
            flagged = flagged_per_node.get(node_idx, set())
            clean_accepted = [s for s in accepted if s not in flagged]

            # Need minimum reference set to compute meaningful stats
            if len(clean_accepted) < 2:
                continue

            # Compute stats on clean accepted set (our reference for "honest")
            stats = self._compute_sender_stats(node_idx, clean_accepted, ctx.neighbor_models)

            if not stats.robust_center:
                continue

            # Check each rejected sender
            rescued = set()
            for sender_id in rejected:
                # Only rescue if trust is sufficient
                if self.trust.get_trust(sender_id) < self.config.phase2_trust_threshold:
                    continue

                # Check if sender's update is consistent with honest consensus
                if self._is_sender_consistent(sender_id, stats, node_idx, ctx.neighbor_models):
                    rescued.add(sender_id)
                    self._record_flag(
                        node_idx=node_idx,
                        sender_id=sender_id,
                        stats=stats,
                        phase=2,
                        action="rescued",
                    )

            if rescued:
                rescued_per_node[node_idx] = rescued

        return rescued_per_node

    # -------------------------------------------------------------------------
    # Phase 3: Re-aggregate
    # -------------------------------------------------------------------------

    def _phase3_reaggregate(
        self,
        ctx: RoundContext,
        flagged_per_node: Dict[int, Set[int]],
        rescued_per_node: Dict[int, Set[int]],
    ) -> Set[int]:
        """Re-aggregate nodes with corrected neighbor sets.

        For each node that had flagged or rescued neighbors:
        1. Build clean neighbor set = (original accepted - flagged) + rescued
        2. Collect models for clean set
        3. Re-aggregate using aggregator-appropriate formula
        4. Update node state and context

        Returns:
            Set of node indices that were modified
        """
        modified_nodes: Set[int] = set()

        for node_idx in range(len(self.nodes)):
            node_flagged = flagged_per_node.get(node_idx, set())
            node_rescued = rescued_per_node.get(node_idx, set())

            # Skip if no changes to neighbor set
            if not node_flagged and not node_rescued:
                continue

            # Build clean neighbor set
            original_accepted = set(ctx.accepted_per_node[node_idx])
            clean_set = (original_accepted - node_flagged) | node_rescued

            if len(clean_set) < 1:
                continue

            # Collect models for clean set
            clean_models = []
            for sender_id in clean_set:
                model = ctx.neighbor_models.get((node_idx, sender_id))
                if model is not None:
                    clean_models.append(model)

            if not clean_models:
                continue

            # Re-aggregate using appropriate formula
            new_state = self._reaggregate_with_formula(
                node_idx, clean_models, ctx.pre_agg_states[node_idx]
            )

            # Update node's actual model state
            self.nodes[node_idx].set_model_state(new_state)
            ctx.current_states[node_idx] = new_state

            # Update accepted/rejected lists for metrics
            ctx.accepted_per_node[node_idx] = list(clean_set)
            ctx.rejected_per_node[node_idx] = [
                s for s in ctx.rejected_per_node[node_idx] if s not in node_rescued
            ] + list(node_flagged)

            modified_nodes.add(node_idx)

        return modified_nodes

    # -------------------------------------------------------------------------
    # Phase 4: Update Trust
    # -------------------------------------------------------------------------

    def _phase4_update_trust(
        self,
        flagged_per_node: Dict[int, Set[int]],
        rescued_per_node: Dict[int, Set[int]],
    ) -> None:
        """Update trust scores based on flags and rescues.

        - Penalize all flagged nodes (they're suspected attackers)
        - Boost all rescued nodes (they're confirmed honest)
        - Decay all scores toward neutral over time
        """
        # Collect all unique flagged nodes
        all_flagged: Set[int] = set()
        for node_set in flagged_per_node.values():
            all_flagged.update(node_set)

        for node_id in all_flagged:
            self.trust.penalize(node_id)

        # Collect all unique rescued nodes
        all_rescued: Set[int] = set()
        for node_set in rescued_per_node.values():
            all_rescued.update(node_set)

        for node_id in all_rescued:
            self.trust.boost(node_id)

        # Decay all scores toward neutral
        self.trust.decay_towards_neutral()

    # -------------------------------------------------------------------------
    # Core Statistics Functions
    # -------------------------------------------------------------------------

    def _compute_sender_stats(
        self,
        node_idx: int,
        sender_ids: List[int],
        neighbor_models: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    ) -> SenderStats:
        """Compute robust statistics on what senders SENT to this node.

        Uses coordinate-wise median as robust center (resistant to outliers).
        Uses MAD (Median Absolute Deviation) for spread estimation.

        Args:
            node_idx: The receiving node
            sender_ids: List of sender IDs to analyze
            neighbor_models: Map of (receiver, sender) -> model state dict

        Returns:
            SenderStats with robust center, distances, and outlier threshold
        """
        # Collect models that were sent
        models = []
        valid_senders = []
        for sender_id in sender_ids:
            model = neighbor_models.get((node_idx, sender_id))
            if model is not None:
                models.append(model)
                valid_senders.append(sender_id)

        # Need at least 2 models for meaningful statistics
        if len(models) < 2:
            return SenderStats(
                robust_center={},
                distances={},
                median_dist=0.0,
                mad=1.0,
                outlier_threshold=float("inf"),
            )

        # Compute coordinate-wise median as robust center
        keys = list(models[0].keys())
        robust_center: Dict[str, torch.Tensor] = {}

        for key in keys:
            stacked = torch.stack([m[key].float() for m in models])
            robust_center[key] = stacked.median(dim=0).values

        # Compute distance from each sender to robust center
        distances: Dict[int, float] = {}
        for i, sender_id in enumerate(valid_senders):
            model = models[i]
            dist = 0.0
            for key in keys:
                diff = model[key].float() - robust_center[key]
                dist += torch.sum(diff * diff).item()
            distances[sender_id] = float(np.sqrt(dist))

        # Compute MAD (Median Absolute Deviation) for robust spread estimate
        dist_values = list(distances.values())
        median_dist = float(np.median(dist_values))
        mad = float(np.median([abs(d - median_dist) for d in dist_values]))

        # Robust threshold: median + k * scaled_MAD
        # Scale factor 1.4826 makes MAD consistent with std for normal distributions
        # k=2.5 corresponds to ~99% coverage for normal data
        scaled_mad = 1.4826 * mad if mad > 1e-9 else 0.1
        outlier_threshold = median_dist + self._outlier_k * scaled_mad

        return SenderStats(
            robust_center=robust_center,
            distances=distances,
            median_dist=median_dist,
            mad=mad,
            outlier_threshold=outlier_threshold,
        )

    def _is_sender_outlier(
        self,
        sender_id: int,
        stats: SenderStats,
        node_idx: int,
        all_senders: List[int],
        neighbor_models: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
        receiver_model: Dict[str, torch.Tensor] = None,
    ) -> bool:
        """Check if sender's update is a statistical outlier.

        Uses multiple detection signals with layered thresholds:
        1. Distance-based + Direction-based (standard, high confidence)
        2. Extreme direction alone (very low score = obvious attack)
        3. Strong direction + high magnitude (backup for when distance fails)

        ENHANCED: Uses tighter thresholds for previously-accepted nodes.
        An anomaly from an accepted node is more significant because it
        represents a behavioral CHANGE, not baseline behavior.

        Args:
            sender_id: The sender to check
            stats: Pre-computed sender statistics
            node_idx: The receiving node
            all_senders: List of all sender IDs to this node
            neighbor_models: Map of (receiver, sender) -> model state dict
            receiver_model: The receiving node's pre-aggregation model

        Returns:
            True if sender shows attack signature
        """
        # Acceptance-aware threshold adjustment
        record = self.acceptance.get_record(sender_id)
        acceptance_multiplier = 1.0

        if record.status == AcceptanceStatus.ACCEPTED:
            # Accepted node: use TIGHTER thresholds (heightened scrutiny)
            # A deviation from an accepted node is more significant
            acceptance_multiplier = self._post_acceptance_sensitivity
        elif record.status == AcceptanceStatus.SUSPICIOUS:
            # Already suspicious: use even tighter thresholds
            acceptance_multiplier = self._post_acceptance_sensitivity * 1.5

        # Apply multiplier to effective thresholds (higher multiplier = tighter)
        effective_direction_threshold = self._direction_threshold / acceptance_multiplier
        effective_extreme_threshold = 0.10 / acceptance_multiplier
        effective_strong_threshold = 0.20 / acceptance_multiplier
        effective_magnitude_threshold = 3.0 / acceptance_multiplier

        # Check 1: Distance-based outlier
        is_distance_outlier = False
        distance = stats.distances.get(sender_id, 0.0)
        if sender_id in stats.distances:
            is_distance_outlier = distance > stats.outlier_threshold

        # Check 2: Direction-based outlier (catches directed attacks)
        # Returns (score, relative_magnitude) for layered detection
        direction_score, rel_magnitude = self._compute_direction_score(
            sender_id, node_idx, all_senders, neighbor_models, receiver_model
        )
        is_direction_outlier = direction_score < effective_direction_threshold

        if self._debug and node_idx == 2:  # Only print for one node to reduce spam
            status_str = record.status.value[:3]
            print(f"      [Debug] sender={sender_id} status={status_str} "
                  f"dist_outlier={is_distance_outlier} "
                  f"dir_score={direction_score:.3f} rel_mag={rel_magnitude:.3f} "
                  f"dir_outlier={is_direction_outlier}")

        # Layered flagging thresholds (from most to least confident):
        #
        # 1. Standard: BOTH distance AND direction outlier
        standard_outlier = is_distance_outlier and is_direction_outlier

        # 2. Extreme direction WITH high magnitude
        #    Low score alone is not enough - honest victims can also have low scores
        #    after FedAvg pollution. Attackers have BOTH low score AND high magnitude.
        extreme_direction = (
            direction_score < effective_extreme_threshold
            and rel_magnitude > 5.0 / acceptance_multiplier
        )

        # 3. Strong direction + magnitude: direction shows attack signature AND large deviation
        #    Uses magnitude threshold to distinguish attackers (who SEND concentrated
        #    attacks) from victims (who RECEIVE diluted attacks through aggregation)
        strong_direction_with_magnitude = (
            direction_score < effective_strong_threshold
            and rel_magnitude > effective_magnitude_threshold
        )

        # 4. Low trust amplification: once a node is suspected, lower thresholds
        #    This helps catch attackers who evaded detection in earlier rounds
        sender_trust = self.trust.get_trust(sender_id)
        low_trust_suspect = (
            sender_trust < 0.35
            and direction_score < 0.25
            and rel_magnitude > 2.0
        )

        # 5. Post-acceptance anomaly: accepted node with any significant deviation
        post_acceptance_anomaly = (
            record.status in (AcceptanceStatus.ACCEPTED, AcceptanceStatus.SUSPICIOUS)
            and direction_score < 0.30
            and rel_magnitude > 2.0
        )

        return (
            standard_outlier
            or extreme_direction
            or strong_direction_with_magnitude
            or low_trust_suspect
            or post_acceptance_anomaly
        )

    def _compute_direction_score(
        self,
        sender_id: int,
        node_idx: int,
        all_senders: List[int],
        neighbor_models: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
        receiver_model: Dict[str, torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """Detect attacks by comparing sender deviation from PEER MEDIAN (excluding self).

        Key insight: Directed attacks are anti-correlated with the honest average.
        Using PEER MEDIAN (excluding sender) as reference prevents:
        1. Attackers from polluting their own reference
        2. Non-IID causing honest pairs to look anti-correlated

        Detection uses ROBUST STATISTICS:
        - Compute magnitude and direction for ALL senders
        - Flag senders that are outliers in BOTH dimensions
        - Use z-score based thresholds rather than fixed values

        Returns:
            Tuple of (score, relative_magnitude):
            - score in [0, 1]: ~1.0 = consistent, <0.3 = attack signature
            - relative_magnitude: deviation_norm / reference_norm
        """
        sender_model = neighbor_models.get((node_idx, sender_id))
        if sender_model is None:
            return 1.0, 0.0

        # Build reference from peers EXCLUDING this sender
        peer_ids = [s for s in all_senders if s != sender_id]
        peer_models_dict = {
            s: neighbor_models.get((node_idx, s))
            for s in peer_ids
            if neighbor_models.get((node_idx, s)) is not None
        }

        if len(peer_models_dict) < 1:
            return 1.0, 0.0

        keys = list(sender_model.keys())

        # ROBUST REFERENCE: Use receiver's model as anchor to identify attacker updates
        # Attackers' updates deviate far from the receiver's accumulated honest model
        if len(peer_models_dict) > 2 and receiver_model is not None:
            # Use receiver's model as anchor
            receiver_vec = torch.cat([receiver_model[k].float().flatten() for k in keys])

            # Compute distance of each peer from receiver
            peer_dists = {}
            for s, m in peer_models_dict.items():
                peer_vec = torch.cat([m[k].float().flatten() for k in keys])
                diff = peer_vec - receiver_vec
                peer_dists[s] = torch.sqrt(torch.sum(diff * diff)).item()

            # Select K closest peers (smallest distance to receiver)
            K = max(2, (len(peer_models_dict) + 1) // 2)  # Use ~half the peers
            sorted_peers = sorted(peer_dists.items(), key=lambda x: x[1])
            closest_ids = [s for s, _ in sorted_peers[:K]]
            peer_models = [peer_models_dict[s] for s in closest_ids]
        else:
            peer_models = list(peer_models_dict.values())

        # Compute peer median as reference (from closest peers only)
        if len(peer_models) == 1:
            ref_flat = [peer_models[0][key].float().flatten() for key in keys]
        else:
            ref_flat = []
            for key in keys:
                stacked = torch.stack([m[key].float() for m in peer_models])
                ref = stacked.median(dim=0).values
                ref_flat.append(ref.flatten())

        ref_vec = torch.cat(ref_flat)
        sender_vec = torch.cat([sender_model[key].float().flatten() for key in keys])

        # Compute deviation from peer median
        deviation = sender_vec - ref_vec

        # MAGNITUDE: how far is sender from peers?
        dev_norm = torch.sqrt(torch.sum(deviation * deviation)).item()
        ref_norm = torch.sqrt(torch.sum(ref_vec * ref_vec)).item()
        relative_magnitude = dev_norm / max(ref_norm, 1e-9)

        # DIRECTION: is deviation anti-correlated with peer median?
        dot_product = torch.sum(deviation * ref_vec).item()
        cosine = dot_product / (dev_norm * ref_norm) if dev_norm > 1e-9 and ref_norm > 1e-9 else 0.0

        # ADAPTIVE SCORING
        # Attack signature: negative cosine (opposite direction) + magnitude above noise
        # Honest: cosine around 0 or positive, lower magnitude

        # Direction risk: negative cosine is suspicious
        # Scale: 0 for cosine >= 0, up to 1 for cosine = -1
        dir_risk = max(0.0, -cosine)

        # Magnitude risk: deviation above typical non-IID noise floor
        # Small deviations (~0.3) normal in non-IID, larger ones suspicious
        noise_floor = 0.3
        mag_risk = max(0.0, (relative_magnitude - noise_floor) / (2.0 - noise_floor))

        # Combined risk: direction is primary signal, magnitude amplifies
        # Use multiplicative combination: requires BOTH signals for high risk
        combined_risk = dir_risk * (0.5 + mag_risk)

        # Score: lower = more suspicious
        score = max(0.0, min(1.0, 1.0 - combined_risk))

        if self._debug:
            print(f"    [Debug] Sender {sender_id} -> node {node_idx}: "
                  f"rel_mag={relative_magnitude:.3f}, cosine={cosine:.3f}, "
                  f"mag_risk={mag_risk:.3f}, dir_risk={dir_risk:.3f}, score={score:.3f}")

        return score, relative_magnitude

    def _is_sender_consistent(
        self,
        sender_id: int,
        stats: SenderStats,
        node_idx: int,
        neighbor_models: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    ) -> bool:
        """Check if sender's update is consistent with honest consensus.

        A sender is consistent if their update is close to the robust center
        (within the outlier threshold).

        Args:
            sender_id: The sender to check
            stats: Statistics from the clean accepted set
            node_idx: The receiving node
            neighbor_models: Map of (receiver, sender) -> model state dict

        Returns:
            True if sender's update is within normal bounds
        """
        model = neighbor_models.get((node_idx, sender_id))
        if model is None or not stats.robust_center:
            return False

        # Compute distance to robust center
        dist = 0.0
        for key in stats.robust_center.keys():
            diff = model[key].float() - stats.robust_center[key]
            dist += torch.sum(diff * diff).item()
        dist = float(np.sqrt(dist))

        # Consistent if within threshold (not an outlier)
        return dist <= stats.outlier_threshold

    def _evaluate_model(
        self,
        node_idx: int,
        model_state: Dict[str, torch.Tensor],
    ) -> float:
        """Evaluate a model state on a node's local test set.

        Temporarily sets the model state, evaluates, then restores original.

        Args:
            node_idx: The node whose test set to use
            model_state: The model state to evaluate

        Returns:
            Accuracy on the node's test set
        """
        node = self.nodes[node_idx]
        original_state = node.get_model_state()
        node.set_model_state(model_state)
        acc, _ = node.evaluate()
        node.set_model_state(original_state)
        return acc

   
    # Re-aggregation Helpers
    

    def _reaggregate_with_formula(
        self,
        node_idx: int,
        models: List[Dict[str, torch.Tensor]],
        pre_agg_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply aggregator-appropriate formula for re-aggregation.

        Different aggregators use different formulas:
        - BALANCE/UBAR: alpha-blend average with pre-aggregation state
        - Krum: select model closest to centroid
        - MultiKrum: average of k closest models
        - TrimmedMean: coordinate-wise trimmed mean
        - Median: coordinate-wise median
        - FedAvg: simple average

        Args:
            node_idx: The node being re-aggregated
            models: List of clean neighbor models
            pre_agg_state: Node's state before aggregation

        Returns:
            New aggregated state
        """
        agg_name = self.config.aggregation.lower()

        if len(models) == 0:
            return pre_agg_state

        if len(models) == 1:
            # Single model: alpha-blend with pre-agg state
            alpha = getattr(self.config, "balance_alpha", 0.5)
            return self._alpha_blend(pre_agg_state, models[0], alpha)

        if agg_name in ["balance", "ubar"]:
            # Alpha-blend average with pre-agg state
            alpha = getattr(self.config, "balance_alpha", 0.5)
            avg = average_state_dicts(models)
            return self._alpha_blend(pre_agg_state, avg, alpha)

        elif agg_name == "krum":
            # Select model closest to centroid
            return self._krum_select(models)

        elif agg_name == "multikrum":
            # Average of k closest models
            k = max(1, len(models) - 2)
            return self._multi_krum_select(models, k)

        elif agg_name in ["trimmedmean", "trimmed_mean"]:
            # Coordinate-wise trimmed mean
            return self._trimmed_mean(models, trim_ratio=0.1)

        elif agg_name == "median":
            # Coordinate-wise median
            return self._coordinate_median(models)

        else:
            # FedAvg or unknown: simple average
            return average_state_dicts(models)

    def _alpha_blend(
        self,
        own_state: Dict[str, torch.Tensor],
        neighbor_avg: Dict[str, torch.Tensor],
        alpha: float,
    ) -> Dict[str, torch.Tensor]:
        """Blend states using BALANCE formula: alpha * own + (1 - alpha) * neighbors."""
        result = {}
        for key in own_state.keys():
            result[key] = (alpha * own_state[key].float() + (1 - alpha) * neighbor_avg[key].float()).to(
                own_state[key].dtype
            )
        return result

    def _krum_select(self, models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Select the model closest to the centroid (Krum aggregation)."""
        if len(models) == 1:
            return models[0]

        # Compute centroid
        centroid = average_state_dicts(models)

        # Find model closest to centroid
        min_dist = float("inf")
        best_model = models[0]

        for model in models:
            dist = 0.0
            for key in model.keys():
                diff = model[key].float() - centroid[key]
                dist += torch.sum(diff * diff).item()

            if dist < min_dist:
                min_dist = dist
                best_model = model

        return best_model

    def _multi_krum_select(
        self, models: List[Dict[str, torch.Tensor]], k: int
    ) -> Dict[str, torch.Tensor]:
        """Average the k models closest to the centroid (Multi-Krum)."""
        if len(models) <= k:
            return average_state_dicts(models)

        # Compute centroid
        centroid = average_state_dicts(models)

        # Compute distances to centroid
        distances = []
        for i, model in enumerate(models):
            dist = 0.0
            for key in model.keys():
                diff = model[key].float() - centroid[key]
                dist += torch.sum(diff * diff).item()
            distances.append((dist, i))

        # Sort by distance and take k closest
        distances.sort(key=lambda x: x[0])
        selected_indices = [idx for _, idx in distances[:k]]
        selected_models = [models[i] for i in selected_indices]

        return average_state_dicts(selected_models)

    def _trimmed_mean(
        self, models: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise trimmed mean (remove extreme values)."""
        if len(models) < 3:
            return average_state_dicts(models)

        n = len(models)
        trim_count = max(1, int(n * trim_ratio))

        result = {}
        for key in models[0].keys():
            stacked = torch.stack([m[key].float() for m in models])  # Shape: (n, ...)

            # Sort along first dimension and trim extremes
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_count : n - trim_count]

            if trimmed.shape[0] > 0:
                result[key] = trimmed.mean(dim=0).to(models[0][key].dtype)
            else:
                result[key] = stacked.mean(dim=0).to(models[0][key].dtype)

        return result

    def _coordinate_median(
        self, models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation."""
        result = {}
        for key in models[0].keys():
            stacked = torch.stack([m[key].float() for m in models])
            result[key] = stacked.median(dim=0).values.to(models[0][key].dtype)
        return result

   
    # Record Keeping


    def _record_flag(
        self,
        node_idx: int,
        sender_id: int,
        stats: SenderStats,
        phase: int,
        action: str,
        aggregator_flagged: bool = False,
        post_acceptance: bool = False,
        behavioral_signals: Optional[Dict] = None,
    ) -> None:
        """Record a flag or rescue action for metrics tracking.

        Args:
            node_idx: The receiving node
            sender_id: The sender being flagged/rescued
            stats: Statistics used for the decision
            phase: Phase number (1 for flag, 2 for rescue)
            action: "flagged" or "rescued"
            aggregator_flagged: True if flagged by aggregator (e.g., BALANCE rejection)
            post_acceptance: True if this is a post-acceptance attack detection
            behavioral_signals: Optional dict with trust_drop, volatility, trend, etc.
        """
        trust_before = self.trust.get_trust(sender_id)
        acceptance_record = self.acceptance.get_record(sender_id)

        record = {
            "node_id": sender_id,
            "target_node": node_idx,
            "phase": phase,
            "action": action,
            "trust_before": trust_before,
            "aggregator_flagged": aggregator_flagged,
            "post_acceptance": post_acceptance,
            "acceptance_status": acceptance_record.status.value,
            "signals": {
                "distance": stats.distances.get(sender_id, 0.0),
                "median_dist": stats.median_dist,
                "mad": stats.mad,
                "outlier_threshold": stats.outlier_threshold,
                "is_outlier": stats.distances.get(sender_id, 0.0) > stats.outlier_threshold,
            },
        }

        # Add behavioral signals if this is a post-acceptance detection
        if behavioral_signals:
            record["behavioral_signals"] = behavioral_signals

        self.flags.append(record)


class NoOpVerificationLayer:
    """No-op verification layer for when verification is disabled.

    Provides the same interface but performs no verification.
    """

    def __init__(self, config: SimulationConfig = None, nodes=None, graph=None):
        num_nodes = config.num_nodes if config else 0
        self.trust = type(
            "FakeTrust",
            (),
            {"trust_scores": [0.5] * num_nodes, "get_trust": lambda self, i: 0.5},
        )()
        self.flags: List[dict] = []
        self.flags_per_round: List[List[dict]] = []
        self.time_per_round: List[float] = []

    def run_verification(self, ctx: RoundContext) -> bool:
        """No-op: return False (no changes made)."""
        self.flags_per_round.append([])
        self.time_per_round.append(0.0)
        return False
