"""Coordinate-wise Trimmed Mean and Median aggregators.

References:
    Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal
    Statistical Rates" (ICML 2018).
"""

from typing import Dict, List, Tuple

import torch

from dfl.aggregators.base import BaseAggregator
from dfl.config import SimulationConfig


class TrimmedMeanAggregator(BaseAggregator):
    """Coordinate-wise trimmed mean: trims the top and bottom beta
    fraction of values for each parameter coordinate before averaging.

    This defence is effective against attacks that shift individual
    coordinates to extreme values but is vulnerable to attacks like
    ALIE that stay within the trimmed range.
    """

    def __init__(self, node_id: int, config: SimulationConfig, **kwargs):
        super().__init__(node_id, config)
        self.beta = config.trimmed_mean_beta
        self.stats = {"trimmed_count": 0, "total_aggregations": 0}

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
        n = len(all_models)
        trim_count = max(1, int(self.beta * n))

        aggregated = {}
        for key in own_model.keys():
            stacked = torch.stack([m[key].float() for m in all_models])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            # Trim top and bottom
            trimmed = sorted_vals[trim_count:n - trim_count]
            if trimmed.numel() == 0:
                # If too few models, fall back to full mean
                aggregated[key] = stacked.mean(dim=0).to(own_model[key].dtype)
            else:
                aggregated[key] = trimmed.mean(dim=0).to(own_model[key].dtype)

        self.stats["total_aggregations"] += 1
        self.stats["trimmed_count"] += trim_count * 2

        # Trimmed mean accepts all (it trims per-coordinate, not per-model)
        return aggregated, list(neighbor_indices), []


class CoordinateMedianAggregator(BaseAggregator):
    """Coordinate-wise median: takes the median of each parameter
    coordinate across all models.

    More robust than trimmed mean against larger fractions of Byzantine
    nodes, but can introduce bias in non-symmetric distributions.
    """

    def __init__(self, node_id: int, config: SimulationConfig, **kwargs):
        super().__init__(node_id, config)
        self.stats = {"total_aggregations": 0}

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

        aggregated = {}
        for key in own_model.keys():
            stacked = torch.stack([m[key].float() for m in all_models])
            aggregated[key] = stacked.median(dim=0).values.to(own_model[key].dtype)

        self.stats["total_aggregations"] += 1

        # Median uses all models (coordinate-wise operation)
        return aggregated, list(neighbor_indices), []
