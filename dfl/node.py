"""Federated node: a single participant in the decentralised network."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dfl.aggregators import create_aggregator
from dfl.config import SimulationConfig


class FederatedNode:
    """Represents a single node in the federated network.

    Each node owns a local model, trains it on its private data, and
    aggregates its model with neighbours using the configured algorithm.
    """

    def __init__(
        self,
        node_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: SimulationConfig,
        device: torch.device,
        total_rounds: int,
    ):
        self.node_id = node_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Create the aggregator for this node
        self.aggregator = create_aggregator(
            config.aggregation,
            node_id,
            config,
            total_rounds=total_rounds,
            train_loader=train_loader,
            device=device,
            model_template=model,
        )

    def train_local(self):
        """Train the model on local data for the configured number of epochs."""
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.config.learning_rate, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate the model on local test data.

        Returns a tuple of (accuracy, average_loss).
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

                _, predicted = torch.max(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / max(1, total)
        avg_loss = total_loss / max(1, total)
        if not np.isfinite(avg_loss):
            avg_loss = float("inf")
        return accuracy, avg_loss

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the model's state dict."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_model_state(self, state: Dict[str, torch.Tensor]):
        """Load a state dict into the model."""
        self.model.load_state_dict(state, strict=False)
