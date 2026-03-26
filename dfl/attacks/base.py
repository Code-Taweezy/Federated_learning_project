#Abstract base class for all Byzantine attack strategies.

import random
from typing import Dict, List, Optional, Set

import torch

from dfl.config import SimulationConfig
from dfl.utils import average_state_dicts


class BaseByzantineAttacker:
    """Base class that selects compromised nodes and defines the attack interface.

    Subclasses must implement ``_craft_attack`` to define how
    malicious model updates are generated from the honest neighbours' updates.

    Supports delayed attacks via attack_start_round: attackers behave honestly
    until that round, then turn malicious. This enables post-acceptance attack
    simulation where nodes build trust before attacking.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_compromised = int(config.num_nodes * config.attack_ratio)
        self.attack_start_round = config.attack_start_round
        self._current_round = 0

        # Randomly select which nodes are compromised using a fixed seed
        random.seed(config.seed)
        self.compromised_nodes: Set[int] = set(
            random.sample(range(config.num_nodes), self.num_compromised)
        )
        print(
            f"Compromised {len(self.compromised_nodes)} nodes: "
            f"{sorted(self.compromised_nodes)}"
        )
        if self.attack_start_round > 0:
            print(f"Attackers will begin attacking at round {self.attack_start_round}")

    def set_current_round(self, round_num: int) -> None:
        """Update the current round number for delayed attack activation."""
        self._current_round = round_num

    def is_compromised(self, node_id: int) -> bool:
        """Return True if the given node is Byzantine."""
        return node_id in self.compromised_nodes

    def is_active(self, node_id: int) -> bool:
        """Check if attack is active for this node in current round.

        Returns False before attack_start_round (node behaves honestly).
        """
        if not self.is_compromised(node_id):
            return False
        return self._current_round >= self.attack_start_round

    def craft_malicious_update(
        self,
        honest_updates: List[Dict[str, torch.Tensor]],
        node_id: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Create a malicious model update, or None if attack not active.

        Returns None before attack_start_round, signaling the node should
        behave honestly (use average of honest updates instead).
        """
        if not self.is_active(node_id):
            return None  # Behave honestly before attack_start_round

        return self._craft_attack(honest_updates, node_id)

    def _craft_attack(
        self,
        honest_updates: List[Dict[str, torch.Tensor]],
        node_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Actual attack implementation - must be overridden by subclasses."""
        raise NotImplementedError
