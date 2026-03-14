"""Abstract base class for all Byzantine attack strategies."""

import random
from typing import Dict, List, Set

import torch

from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts


class BaseByzantineAttacker:
    """Base class that selects compromised nodes and defines the attack interface.

    Subclasses must implement ``craft_malicious_update`` to define how
    malicious model updates are generated from the honest neighbours' updates.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_compromised = int(config.num_nodes * config.attack_ratio)

        # Randomly select which nodes are compromised using a fixed seed
        random.seed(config.seed)
        self.compromised_nodes: Set[int] = set(
            random.sample(range(config.num_nodes), self.num_compromised)
        )
        print(
            f"Compromised {len(self.compromised_nodes)} nodes: "
            f"{sorted(self.compromised_nodes)}"
        )

    def is_compromised(self, node_id: int) -> bool:
        """Return True if the given node is Byzantine."""
        return node_id in self.compromised_nodes

    def craft_malicious_update(
        self,
        honest_updates: List[Dict[str, torch.Tensor]],
        node_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Create a malicious model update.

        Must be implemented by every concrete attack strategy.
        """
        raise NotImplementedError
