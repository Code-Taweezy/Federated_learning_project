"""FedAvg aggregator: simple averaging of all neighbours with no filtering."""

from typing import Dict, List, Tuple

import torch

from dfl.aggregators.base import BaseAggregator
from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts


class FedAvgAggregator(BaseAggregator):
    """Standard Federated Averaging: accepts all neighbours equally."""

    def __init__(self, node_id: int, config: SimulationConfig, **kwargs):
        super().__init__(node_id, config)
        self.stats = {"accepted": 0, "total": 0}

    def aggregate(
        self,
        own_model: Dict[str, torch.Tensor],
        neighbor_models: List[Dict[str, torch.Tensor]],
        neighbor_indices: List[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        all_models = [own_model] + neighbor_models
        self.stats["total"] += len(neighbor_models)
        self.stats["accepted"] += len(neighbor_models)

        accepted = list(neighbor_indices) if neighbor_indices is not None else []
        return average_state_dicts(all_models), accepted, []
