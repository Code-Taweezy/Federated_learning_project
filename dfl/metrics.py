"""Per-round and cumulative detection metrics for Byzantine fault detection.

Tracks true positives, false positives, true negatives, false negatives,
and derives precision, recall, F1, attack success rate (ASR), and
time-to-detection from those counts.

Extended with post-acceptance attack detection metrics:
- Tracks which attackers achieved "accepted" status before attacking
- Detection latency after attack begins
- Pre-acceptance vs post-acceptance detection rates
"""

from typing import Dict, List, Optional, Set


class MetricsTracker:
    """Tracks detection performance across training rounds.

    This replaces the scattered confusion-matrix accumulators that were
    previously embedded in the simulator. It provides both per-round
    and cumulative metrics, and can produce a summary dict suitable for
    JSON serialisation.

    Extended with post-acceptance attack metrics for research on detecting
    nodes that behave honestly to build trust, then turn malicious.
    """

    def __init__(self, num_nodes: int, compromised_nodes: Set[int]):
        self.num_nodes = num_nodes
        self.compromised_nodes = set(compromised_nodes)
        self.per_round: List[Dict] = []

        # Cumulative confusion matrix
        self.cumulative_tp = 0
        self.cumulative_fp = 0
        self.cumulative_tn = 0
        self.cumulative_fn = 0
        self.detection_time: Optional[int] = None

        # Post-acceptance attack tracking
        self.attackers_achieved_acceptance: Set[int] = set()
        self.attacker_acceptance_rounds: Dict[int, int] = {}  # attacker_id -> round
        self.attacker_attack_start_rounds: Dict[int, int] = {}  # attacker_id -> round
        self.attacker_detection_rounds: Dict[int, int] = {}  # attacker_id -> round

        # Pre vs post acceptance detection counts
        self.pre_acceptance_detections: int = 0
        self.post_acceptance_detections: int = 0

    def record_acceptance(self, node_id: int, round_num: int) -> None:
        """Record when a node achieves acceptance status.

        Args:
            node_id: The node that achieved acceptance
            round_num: The round when acceptance was achieved
        """
        if node_id in self.compromised_nodes:
            self.attackers_achieved_acceptance.add(node_id)
            if node_id not in self.attacker_acceptance_rounds:
                self.attacker_acceptance_rounds[node_id] = round_num

    def record_attack_start(self, node_id: int, round_num: int) -> None:
        """Record when an attacker starts attacking.

        Args:
            node_id: The attacker node
            round_num: The round when attacks begin
        """
        if node_id in self.compromised_nodes:
            if node_id not in self.attacker_attack_start_rounds:
                self.attacker_attack_start_rounds[node_id] = round_num

    def record_detection(
        self,
        node_id: int,
        round_num: int,
        was_accepted: bool,
    ) -> None:
        """Record when an attacker is detected.

        Args:
            node_id: The detected node
            round_num: The round of detection
            was_accepted: Whether the node had achieved acceptance before detection
        """
        if node_id not in self.compromised_nodes:
            return

        # Track first detection round per attacker
        if node_id not in self.attacker_detection_rounds:
            self.attacker_detection_rounds[node_id] = round_num

        # Track pre vs post acceptance detection
        if was_accepted:
            self.post_acceptance_detections += 1
        else:
            self.pre_acceptance_detections += 1

    def update_round(self, round_num: int, flags: List[dict]) -> Dict:
        """Record detection results for a single round.

        ``flags`` is the per-node flag list produced by the anomaly detector,
        where each element has at least ``node_id`` and ``flagged`` keys.

        Returns a dict of per-round metrics.
        """
        if not flags or not self.compromised_nodes:
            self.per_round.append(self._empty_round(round_num))
            return self.per_round[-1]

        round_tp = 0
        round_fp = 0
        round_tn = 0
        round_fn = 0

        for f in flags:
            nid = f["node_id"]
            is_compromised = nid in self.compromised_nodes
            is_flagged = f["flagged"]

            if is_compromised and is_flagged:
                round_tp += 1
                if self.detection_time is None:
                    self.detection_time = round_num
            elif not is_compromised and is_flagged:
                round_fp += 1
            elif is_compromised and not is_flagged:
                round_fn += 1
            else:
                round_tn += 1

        self.cumulative_tp += round_tp
        self.cumulative_fp += round_fp
        self.cumulative_tn += round_tn
        self.cumulative_fn += round_fn

        precision = self._safe_div(round_tp, round_tp + round_fp)
        recall = self._safe_div(round_tp, round_tp + round_fn)
        f1 = self._f1(precision, recall)

        # Attack success rate: fraction of compromised nodes that were NOT flagged
        num_compromised_in_flags = sum(
            1 for f in flags if f["node_id"] in self.compromised_nodes
        )
        num_flagged_compromised = round_tp
        asr = 1.0 - self._safe_div(num_flagged_compromised, num_compromised_in_flags)

        metrics = {
            "round": round_num,
            "tp": round_tp,
            "fp": round_fp,
            "tn": round_tn,
            "fn": round_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "asr": asr,
        }
        self.per_round.append(metrics)
        return metrics

    def get_summary(self) -> Dict:
        """Return a summary dict for the entire simulation."""
        precision = self._safe_div(
            self.cumulative_tp, self.cumulative_tp + self.cumulative_fp
        )
        recall = self._safe_div(
            self.cumulative_tp, self.cumulative_tp + self.cumulative_fn
        )
        f1 = self._f1(precision, recall)

        return {
            "true_positives": self.cumulative_tp,
            "false_positives": self.cumulative_fp,
            "true_negatives": self.cumulative_tn,
            "false_negatives": self.cumulative_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "detection_time": self.detection_time,
        }

    def get_post_acceptance_summary(self) -> Dict:
        """Get summary of post-acceptance attack detection performance.

        Returns dict with:
        - attackers_total: Total number of attackers
        - attackers_achieved_acceptance: Number who achieved accepted status
        - acceptance_rate: Fraction who achieved acceptance
        - pre_acceptance_detections: Detections before acceptance
        - post_acceptance_detections: Detections after acceptance
        - post_acceptance_detection_rate: Fraction of accepted attackers detected
        - mean_detection_latency: Average rounds from attack start to detection
        - detection_latencies: List of individual latencies
        """
        num_attackers = len(self.compromised_nodes)
        num_accepted = len(self.attackers_achieved_acceptance)

        # Detection latency: rounds from attack_start to detection
        detection_latencies = []
        for attacker_id in self.compromised_nodes:
            attack_round = self.attacker_attack_start_rounds.get(attacker_id)
            detect_round = self.attacker_detection_rounds.get(attacker_id)
            if attack_round is not None and detect_round is not None:
                latency = detect_round - attack_round
                if latency >= 0:  # Only count if detected after attack started
                    detection_latencies.append(latency)

        # Post-acceptance detection rate: of accepted attackers, how many detected?
        post_acc_detected = len(
            self.attackers_achieved_acceptance & set(self.attacker_detection_rounds.keys())
        )

        return {
            "attackers_total": num_attackers,
            "attackers_achieved_acceptance": num_accepted,
            "acceptance_rate": self._safe_div(num_accepted, num_attackers),
            "pre_acceptance_detections": self.pre_acceptance_detections,
            "post_acceptance_detections": self.post_acceptance_detections,
            "post_acceptance_detection_rate": self._safe_div(post_acc_detected, num_accepted),
            "mean_detection_latency": (
                sum(detection_latencies) / len(detection_latencies)
                if detection_latencies
                else None
            ),
            "detection_latencies": detection_latencies,
            "acceptance_rounds": dict(self.attacker_acceptance_rounds),
            "attack_start_rounds": dict(self.attacker_attack_start_rounds),
            "detection_rounds": dict(self.attacker_detection_rounds),
        }

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        denom = precision + recall
        return 2 * precision * recall / denom if denom > 0 else 0.0

    @staticmethod
    def _empty_round(round_num: int) -> Dict:
        return {
            "round": round_num,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "asr": 0.0,
        }
