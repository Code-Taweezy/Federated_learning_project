"""UBAR aggregator: two-stage Byzantine-robust aggregation."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dfl.aggregators.base import BaseAggregator
from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts, model_distance


class UBARAggregator(BaseAggregator):
    """Two-stage aggregation that combines distance filtering with performance testing.

    Stage 1 selects the closest rho fraction of neighbours by L2 distance.
    Stage 2 keeps only those whose loss on a sample batch is no worse than
    the node's own model. The result is blended with the node's own model
    using the same alpha parameter as BALANCE.
    """

    def __init__(
        self,
        node_id: int,
        config: SimulationConfig,
        train_loader: DataLoader = None,
        device: torch.device = None,
        model_template: nn.Module = None,
        **kwargs,
    ):
        super().__init__(node_id, config)
        self.train_loader = train_loader
        self.device = device
        self.model_template = model_template
        self.criterion = nn.CrossEntropyLoss()
        self.stats = {
            "accepted": 0,
            "total": 0,
            "stage1_rates": [],
            "stage2_rates": [],
        }

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

        # Stage 1: distance-based filtering (keep closest rho fraction)
        distances = [model_distance(own_model, m) for m in neighbor_models]
        num_select = max(1, int(self.config.ubar_rho * len(neighbor_models)))

        sorted_indices = np.argsort(distances)
        stage1_selected = [neighbor_models[i] for i in sorted_indices[:num_select]]
        stage1_nids = [neighbor_indices[i] for i in sorted_indices[:num_select]]

        stage1_rate = len(stage1_selected) / max(1, len(neighbor_models))
        self.stats["stage1_rates"].append(stage1_rate)

        # Stage 2: performance-based filtering (keep models with loss <= own)
        try:
            sample_batch = next(iter(self.train_loader))
            own_loss = self._compute_loss(own_model, sample_batch)

            stage2_selected = []
            stage2_nids = []
            for model, nid in zip(stage1_selected, stage1_nids):
                model_loss = self._compute_loss(model, sample_batch)
                if model_loss <= own_loss:
                    stage2_selected.append(model)
                    stage2_nids.append(nid)

            if not stage2_selected:
                stage2_selected = [stage1_selected[0]]
                stage2_nids = [stage1_nids[0]]

            stage2_rate = len(stage2_selected) / max(1, len(stage1_selected))
            self.stats["stage2_rates"].append(stage2_rate)

        except StopIteration:
            stage2_selected = stage1_selected
            stage2_nids = stage1_nids
            self.stats["stage2_rates"].append(1.0)

        self.stats["total"] += len(neighbor_models)
        self.stats["accepted"] += len(stage2_selected)

        accepted_indices = list(stage2_nids)
        rejected_indices = [
            nid for nid in neighbor_indices if nid not in accepted_indices
        ]

        # Convex blend with own model
        neighbor_avg = average_state_dicts(stage2_selected)
        aggregated = {}
        for key in own_model.keys():
            aggregated[key] = (
                self.config.balance_alpha * own_model[key]
                + (1 - self.config.balance_alpha) * neighbor_avg[key]
            )

        return aggregated, accepted_indices, rejected_indices

    def _compute_loss(
        self,
        model_state: Dict[str, torch.Tensor],
        batch: Tuple,
    ) -> float:
        """Evaluate loss of a model state dict on a single batch."""
        self.model_template.load_state_dict(model_state, strict=False)
        self.model_template.eval()

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            logits = self.model_template(x)
            loss = self.criterion(logits, y)

        return loss.item()
