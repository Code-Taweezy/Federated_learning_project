"""Per-round and cumulative detection metrics for Byzantine fault detection.

Tracks true positives, false positives, true negatives, false negatives,
and derives precision, recall, F1, attack success rate (ASR), and
time-to-detection from those counts.
"""

from typing import Dict, List, Optional, Set


class MetricsTracker:
    """Tracks detection performance across training rounds.

    This replaces the scattered confusion-matrix accumulators that were
    previously embedded in the simulator. It provides both per-round
    and cumulative metrics, and can produce a summary dict suitable for
    JSON serialisation.
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
