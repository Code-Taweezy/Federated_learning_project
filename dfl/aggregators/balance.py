"""BALANCE aggregator: distance-threshold filtering with weighted aggregation."""

from typing import Dict, List, Tuple

import numpy as np
import torch

from dfl.aggregators.base import BaseAggregator
from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts, model_distance


class BALANCEAggregator(BaseAggregator):
    """Distance-threshold filtering with exponentially decaying threshold.

    Neighbours whose model is further than the adaptive threshold from
    the node's own model are rejected. If every neighbour is rejected,
    the closest one is accepted as a fallback. The final aggregated model
    is a convex combination of the node's own model and the accepted
    neighbours' average.
    """

    def __init__(self, node_id: int, config: SimulationConfig, total_rounds: int = 50, **kwargs):
        super().__init__(node_id, config)
        self.total_rounds = total_rounds
        self.current_round = 0
        self.stats = {
            "accepted": 0,
            "total": 0,
            "acceptance_rates": [],
            "thresholds": [],
        }

    def _compute_threshold(self, own_model: Dict[str, torch.Tensor]) -> float:
        """Compute the adaptive acceptance threshold for this round."""
        own_norm_sq = 0.0
        for param in own_model.values():
            p = param.float()
            own_norm_sq += torch.sum(p * p).item()
        own_norm = float(np.sqrt(own_norm_sq))
        lambda_t = self.current_round / max(1, self.total_rounds)
        return (
            self.config.balance_gamma
            * np.exp(-self.config.balance_kappa * lambda_t)
            * own_norm
        )

    def aggregate(
        self,
        own_model: Dict[str, torch.Tensor],
        neighbor_models: List[Dict[str, torch.Tensor]],
        neighbor_indices: List[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        self.current_round += 1
        if not neighbor_models:
            return own_model, [], []

        threshold = self._compute_threshold(own_model)
        self.stats["thresholds"].append(threshold)

        accepted_models = []
        accepted_indices = []
        rejected_indices = []

        for idx, neighbor_model in enumerate(neighbor_models):
            distance = model_distance(own_model, neighbor_model)
            neighbor_id = (
                neighbor_indices[idx] if neighbor_indices is not None else idx
            )
            if distance <= threshold:
                accepted_models.append(neighbor_model)
                accepted_indices.append(neighbor_id)
            else:
                rejected_indices.append(neighbor_id)

        # Fallback: accept the closest neighbour if all were rejected
        if not accepted_models:
            closest_idx = min(
                range(len(neighbor_models)),
                key=lambda i: model_distance(own_model, neighbor_models[i]),
            )
            closest_id = (
                neighbor_indices[closest_idx]
                if neighbor_indices is not None
                else closest_idx
            )
            accepted_models.append(neighbor_models[closest_idx])
            accepted_indices.append(closest_id)
            if closest_id in rejected_indices:
                rejected_indices.remove(closest_id)

        self.stats["total"] += len(neighbor_models)
        self.stats["accepted"] += len(accepted_models)
        self.stats["acceptance_rates"].append(
            len(accepted_models) / max(1, len(neighbor_models))
        )

        # Convex combination of own model and accepted neighbours' average
        neighbor_avg = average_state_dicts(accepted_models)
        aggregated = {}
        for key in own_model.keys():
            aggregated[key] = (
                self.config.balance_alpha * own_model[key]
                + (1 - self.config.balance_alpha) * neighbor_avg[key]
            )
        return aggregated, accepted_indices, rejected_indices
