"""ALIE (A Little Is Enough) attack: crafts updates just inside the
detection boundary to evade statistical defences.

Reference:
    Baruch et al., "A Little Is Enough: Circumventing Defenses For
    Distributed Learning" (NeurIPS 2019).
"""

from typing import Dict, List

import numpy as np
import torch

from dfl.attacks.base import BaseByzantineAttacker


class ALIEAttacker(BaseByzantineAttacker):
    """Shifts each parameter by z_max standard deviations from the mean.

    For each parameter coordinate, the attack computes the mean and
    standard deviation across honest updates, then shifts the malicious
    value to ``mean - z_max * std``.  The z_max parameter controls how
    aggressive the attack is -- small values are stealthier but less
    damaging, while larger values cause more harm but are easier to
    detect.

    Unlike Gaussian noise or directed attacks, ALIE specifically targets
    coordinate-wise aggregation defences (like trimmed mean or median)
    by staying within statistical norms.
    """

    def _craft_attack(
        self,
        honest_updates: List[Dict[str, torch.Tensor]],
        node_id: int,
    ) -> Dict[str, torch.Tensor]:
        if not honest_updates:
            return {}

        z_max = self.config.alie_z_max

        malicious = {}
        for key in honest_updates[0].keys():
            stacked = torch.stack([u[key].float() for u in honest_updates])
            mu = stacked.mean(dim=0)
            sigma = stacked.std(dim=0)
            # Shift in the negative direction to pull model away from convergence
            malicious[key] = (mu - z_max * sigma).to(honest_updates[0][key].dtype)

        return malicious
