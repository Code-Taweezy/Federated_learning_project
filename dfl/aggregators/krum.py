"""Krum and Multi-Krum aggregators: score-based Byzantine-robust selection.

Reference:
    Blanchard et al., "Machine Learning with Adversaries: Byzantine
    Tolerant Gradient Descent" (NIPS 2017).
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from dfl.aggregators.base import BaseAggregator
from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts, model_distance


class KrumAggregator(BaseAggregator):
    """Selects the single model with the smallest sum of distances to its
    nearest (n - f - 2) neighbours, where f is the number of assumed
    Byzantine nodes.

    The selected model replaces the aggregated output (no averaging).
    This is maximally conservative: only one model is trusted per round.
    """

    def __init__(self, node_id: int, config: SimulationConfig, **kwargs):
        super().__init__(node_id, config)
        self.stats = {"selected_node": [], "krum_scores": []}

    def aggregate(
        self,
        own_model: Dict[str, torch.Tensor],
        neighbor_models: List[Dict[str, torch.Tensor]],
        neighbor_indices: List[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        if not neighbor_models:
            return own_model, [], []

        if neighbor_indices is None:
            neighbor_indices = list(range(len(neighbor_models)))

        all_models = [own_model] + neighbor_models
        all_ids = [self.node_id] + list(neighbor_indices)
        n = len(all_models)

        # f = assumed number of Byzantine nodes among neighbours
        f = int(self.config.attack_ratio * self.config.num_nodes)
        # Number of closest neighbours to consider for scoring
        n_nearest = max(1, n - f - 2)

        # Pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = model_distance(all_models[i], all_models[j])
                dists[i][j] = d
                dists[j][i] = d

        # Krum score: sum of distances to n_nearest closest neighbours
        scores = []
        for i in range(n):
            sorted_dists = sorted(dists[i])
            # sorted_dists[0] is 0 (distance to self), skip it
            score = sum(sorted_dists[1:n_nearest + 1])
            scores.append(score)

        best_idx = int(np.argmin(scores))
        selected_model = all_models[best_idx]
        selected_id = all_ids[best_idx]

        self.stats["selected_node"].append(selected_id)
        self.stats["krum_scores"].append(scores)

        accepted = [selected_id] if selected_id != self.node_id else []
        rejected = [nid for nid in neighbor_indices if nid not in accepted]

        return selected_model, accepted, rejected


class MultiKrumAggregator(BaseAggregator):
    """Selects the k models with the lowest Krum scores and averages them.

    Multi-Krum generalises Krum by accepting multiple models, providing
    a middle ground between the single-selection conservatism of Krum
    and the full averaging of FedAvg.
    """

    def __init__(self, node_id: int, config: SimulationConfig, **kwargs):
        super().__init__(node_id, config)
        self.k = config.krum_multi_k or max(1, config.num_nodes // 2)
        self.stats = {"selected_nodes": [], "krum_scores": []}

    def aggregate(
        self,
        own_model: Dict[str, torch.Tensor],
        neighbor_models: List[Dict[str, torch.Tensor]],
        neighbor_indices: List[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        if not neighbor_models:
            return own_model, [], []

        if neighbor_indices is None:
            neighbor_indices = list(range(len(neighbor_models)))

        all_models = [own_model] + neighbor_models
        all_ids = [self.node_id] + list(neighbor_indices)
        n = len(all_models)

        f = int(self.config.attack_ratio * self.config.num_nodes)
        n_nearest = max(1, n - f - 2)

        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = model_distance(all_models[i], all_models[j])
                dists[i][j] = d
                dists[j][i] = d

        scores = []
        for i in range(n):
            sorted_dists = sorted(dists[i])
            score = sum(sorted_dists[1:n_nearest + 1])
            scores.append(score)

        # Select the k models with the lowest scores
        k = min(self.k, n)
        top_k_indices = sorted(range(n), key=lambda i: scores[i])[:k]

        selected_models = [all_models[i] for i in top_k_indices]
        selected_ids = {all_ids[i] for i in top_k_indices}

        self.stats["selected_nodes"].append(list(selected_ids))
        self.stats["krum_scores"].append(scores)

        aggregated = average_state_dicts(selected_models)
        accepted = [nid for nid in neighbor_indices if nid in selected_ids]
        rejected = [nid for nid in neighbor_indices if nid not in selected_ids]

        return aggregated, accepted, rejected
