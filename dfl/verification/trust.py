"""Trust score management for the verification layer.

Maintains per-node trust scores, drift history, and peer deviation
history across training rounds.
"""

from typing import List

import numpy as np

from dfl.config import SimulationConfig


class TrustManager:
    """Manages trust scores and historical metric buffers for all nodes.

    Trust scores start at ``config.trust_initial`` and are updated each
    round based on verification outcomes. History buffers store the most
    recent ``config.verification_history_window`` values of drift and
    peer deviation for each node, used by the verification phases to
    assess whether a node's behaviour is persistently suspicious.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.trust_scores: List[float] = [config.trust_initial] * config.num_nodes
        self.drift_history: List[List[float]] = [[] for _ in range(config.num_nodes)]
        self.peer_dev_history: List[List[float]] = [[] for _ in range(config.num_nodes)]

        # Tracks how many consecutive rounds a (receiver, sender) pair
        # has looked suspicious, used for requiring sustained evidence
        self.suspicion_counts = {}

    def update_history(self, drifts: List[float], peer_devs: List[float]) -> None:
        """Append the latest drift and peer deviation values to the history buffers."""
        w = self.config.verification_history_window
        for i in range(self.config.num_nodes):
            self.drift_history[i].append(drifts[i])
            if len(self.drift_history[i]) > w:
                self.drift_history[i].pop(0)
            self.peer_dev_history[i].append(peer_devs[i])
            if len(self.peer_dev_history[i]) > w:
                self.peer_dev_history[i].pop(0)

    def penalize(self, node_id: int) -> float:
        """Reduce the trust score for a flagged node and return the new value."""
        self.trust_scores[node_id] = max(
            0.0, self.trust_scores[node_id] - self.config.trust_penalty
        )
        return self.trust_scores[node_id]

    def boost(self, node_id: int) -> float:
        """Increase the trust score for a rescued node and return the new value."""
        self.trust_scores[node_id] = min(
            1.0, self.trust_scores[node_id] + self.config.trust_boost
        )
        return self.trust_scores[node_id]

    def decay_update(self, z_scores: List[float], modified_nodes: set) -> None:
        """Exponential decay update for nodes not modified in Phases 1 or 2.

        Nodes with high z-scores decay toward 0; others decay toward 1.
        """
        beta = self.config.trust_decay
        for i in range(self.config.num_nodes):
            if i in modified_nodes:
                continue
            was_flagged = abs(z_scores[i]) > self.config.z_high
            target = 0.0 if was_flagged else 1.0
            self.trust_scores[i] = beta * self.trust_scores[i] + (1 - beta) * target

    def get_population_means(self):
        """Return the population mean of historical drift and peer deviation."""
        drift_means = [
            sum(h) / len(h) for h in self.drift_history if h
        ]
        peer_means = [
            sum(h) / len(h) for h in self.peer_dev_history if h
        ]
        pop_mean_drift = float(np.mean(drift_means)) if drift_means else 0.0
        pop_mean_peer = float(np.mean(peer_means)) if peer_means else 0.0
        return pop_mean_drift, pop_mean_peer
