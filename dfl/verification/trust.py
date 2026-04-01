"""Trust score management for the verification layer.

Simple, focused trust management: penalize flagged nodes, boost rescued nodes,
and decay all scores towards neutral over time.

Extended with trajectory tracking for post-acceptance attack detection.
"""

from typing import Dict, List, Tuple

import numpy as np

from dfl.config import SimulationConfig


class TrustManager:
    """Manages per-node trust scores for the verification layer.

    Trust scores start at config.trust_initial (typically 0.5) and are updated
    based on verification outcomes:
    - Penalize: decrease trust when a node is flagged as an attacker
    - Boost: increase trust when a node is rescued (confirmed honest)
    - Decay: slowly move all scores toward neutral over time

    Extended with trajectory tracking for behavioral change detection:
    - Track trust history over time
    - Compute volatility (sudden changes indicate behavioral shifts)
    - Compute trend (declining from high trust = post-acceptance signal)
    - Track peak trust (drop from peak indicates attack)
    """

    def __init__(self, config: SimulationConfig):
        self._config = config
        self.trust_scores: List[float] = [config.trust_initial] * config.num_nodes

        # Trust history for trajectory analysis
        self._trust_history: Dict[int, List[Tuple[int, float]]] = {
            i: [(0, config.trust_initial)] for i in range(config.num_nodes)
        }
        self._current_round = 0

    def set_round(self, round_num: int) -> None:
        #Updates current round for history tracking.
        self._current_round = round_num

    def penalize(self, node_id: int) -> float:
        """Decrease trust for a flagged node.

        Args:
            node_id: The node to penalize

        Returns:
            The new trust score after penalty
        """
        self.trust_scores[node_id] = max(
            0.0, self.trust_scores[node_id] - self._config.trust_penalty
        )
        # Record in history
        self._trust_history[node_id].append(
            (self._current_round, self.trust_scores[node_id])
        )
        return self.trust_scores[node_id]

    def boost(self, node_id: int) -> float:
        """Increase trust for a rescued node.

        Args:
            node_id: The node to boost

        Returns:
            The new trust score after boost
        """
        self.trust_scores[node_id] = min(
            1.0, self.trust_scores[node_id] + self._config.trust_boost
        )
        # Record in history
        self._trust_history[node_id].append(
            (self._current_round, self.trust_scores[node_id])
        )
        return self.trust_scores[node_id]

    def decay_towards_neutral(self) -> None:
        """Decay all trust scores towards neutral (initial value).

        Uses exponential moving average: trust = decay * trust + (1-decay) * neutral
        This prevents trust from staying permanently high or low.
        """
        decay = self._config.trust_decay
        neutral = self._config.trust_initial

        for i in range(len(self.trust_scores)):
            self.trust_scores[i] = decay * self.trust_scores[i] + (1 - decay) * neutral

    def is_low_trust(self, node_id: int) -> bool:
        """Check if a node has low trust (worthy of extra scrutiny).

        Args:
            node_id: The node to check

        Returns:
            True if trust is below phase1_trust_threshold
        """
        return self.trust_scores[node_id] < self._config.phase1_trust_threshold

    def get_trust(self, node_id: int) -> float:
        """Get the current trust score for a node.

        Args:
            node_id: The node to query

        Returns:
            Current trust score in [0.0, 1.0]
        """
        return self.trust_scores[node_id]

    # --- Trajectory tracking methods for post-acceptance detection ---

    def get_trust_trajectory(self, node_id: int, window: int = 10) -> List[float]:
        """Get recent trust values for trajectory analysis.

        Args:
            node_id: The node to query
            window: Number of recent values to return

        Returns:
            List of recent trust scores
        """
        history = self._trust_history.get(node_id, [])
        if not history:
            return [self.trust_scores[node_id]]
        recent = history[-window:]
        return [t for _, t in recent]

    def compute_trust_volatility(self, node_id: int, window: int = 10) -> float:
        """Compute standard deviation of recent trust scores.

        High volatility indicates unstable behavior - potential attack signal.

        Args:
            node_id: The node to analyze
            window: Number of recent values to consider

        Returns:
            Standard deviation of trust scores (0.0 if insufficient history)
        """
        trajectory = self.get_trust_trajectory(node_id, window)
        if len(trajectory) < 2:
            return 0.0
        return float(np.std(trajectory))

    def compute_trust_trend(self, node_id: int, window: int = 10) -> float:
        """Compute slope of trust trajectory.

        Negative trend from high trust = post-acceptance attack signal.

        Args:
            node_id: The node to analyze
            window: Number of recent values to consider

        Returns:
            Slope of trust trajectory (negative = declining)
        """
        trajectory = self.get_trust_trajectory(node_id, window)
        if len(trajectory) < 2:
            return 0.0
        x = np.arange(len(trajectory))
        y = np.array(trajectory)
        if np.std(x) < 1e-9:
            return 0.0
        # Linear regression slope
        slope = np.cov(x, y)[0, 1] / np.var(x)
        return float(slope)

    def get_peak_trust(self, node_id: int) -> float:
        """Get the maximum trust this node has ever achieved.

        Args:
            node_id: The node to query

        Returns:
            Maximum historical trust score
        """
        history = self._trust_history.get(node_id, [])
        if not history:
            return self.trust_scores[node_id]
        return max(t for _, t in history)

    def compute_trust_drop(self, node_id: int) -> float:
        """Compute drop from peak trust.

        Large drop from high peak = strong post-acceptance signal.

        Args:
            node_id: The node to analyze

        Returns:
            Difference between peak and current trust (>0 means declined)
        """
        peak = self.get_peak_trust(node_id)
        current = self.trust_scores[node_id]
        return peak - current
