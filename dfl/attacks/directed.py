"""Directed deviation attack: inverts the sign of the honest average."""

from typing import Dict, List

import torch

from dfl.attacks.base import BaseByzantineAttacker
from dfl.utils import average_state_dicts


class DirectedAttacker(BaseByzantineAttacker):
    """Deviates model parameters in the opposite direction of the honest average.

    For each parameter, this attack computes the sign of the honest average
    and then subtracts it (scaled by attack_strength) from the average,
    effectively pushing the model away from the correct direction.
    """

    def _craft_attack(
        self,
        honest_updates: List[Dict[str, torch.Tensor]],
        node_id: int,
    ) -> Dict[str, torch.Tensor]:
        if not honest_updates:
            return {}

        avg_update = average_state_dicts(honest_updates)
        malicious = {}
        for key, param in avg_update.items():
            direction = torch.sign(param)
            malicious[key] = param - self.config.attack_strength * direction
        return malicious
