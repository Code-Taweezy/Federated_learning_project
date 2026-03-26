#Gaussian noise attack: replaces parameters with scaled random noise.

from typing import Dict, List

import numpy as np
import torch

from dfl.attacks.base import BaseByzantineAttacker
from dfl.utils import average_state_dicts


class GaussianAttacker(BaseByzantineAttacker):
    """Replaces model parameters with Gaussian noise scaled by attack_strength.

    The noise magnitude is scaled by attack_strength * sqrt(200), which
    produces updates that are far from any honest model and should be
    easily detectable by distance-based defences.
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
            noise = (
                torch.randn_like(param.float())
                * self.config.attack_strength
                * np.sqrt(200.0)
            )
            malicious[key] = noise.to(param.dtype)
        return malicious
