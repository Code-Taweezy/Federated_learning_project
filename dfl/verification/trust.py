"""Trust score management for the verification layer.

Simple, focused trust management: penalize flagged nodes, boost rescued nodes,
and decay all scores towards neutral over time.
"""

from typing import List

from dfl.config import SimulationConfig


class TrustManager:
    """Manages per-node trust scores for the verification layer.

    Trust scores start at config.trust_initial (typically 0.5) and are updated
    based on verification outcomes:
    - Penalize: decrease trust when a node is flagged as an attacker
    - Boost: increase trust when a node is rescued (confirmed honest)
    - Decay: slowly move all scores toward neutral over time
    """

    def __init__(self, config: SimulationConfig):
        self._config = config
        self.trust_scores: List[float] = [config.trust_initial] * config.num_nodes

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
