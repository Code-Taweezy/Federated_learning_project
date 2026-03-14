"""Abstract base class for all aggregation algorithms."""

from typing import Dict, List, Tuple

import torch

from dfl.config import SimulationConfig


class BaseAggregator:
    """Interface that every aggregation algorithm must implement.

    The ``aggregate`` method receives the node's own model plus its
    neighbours' models and returns the aggregated result along with
    lists of which neighbours were accepted or rejected.
    """

    def __init__(self, node_id: int, config: SimulationConfig, **kwargs):
        self.node_id = node_id
        self.config = config
        self.stats: Dict = {}

    def aggregate(
        self,
        own_model: Dict[str, torch.Tensor],
        neighbor_models: List[Dict[str, torch.Tensor]],
        neighbor_indices: List[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        """Aggregate own model with neighbours.

        Returns:
            A tuple of (aggregated_model, accepted_indices, rejected_indices).
        """
        raise NotImplementedError
