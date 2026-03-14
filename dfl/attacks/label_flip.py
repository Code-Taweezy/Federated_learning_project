"""Label-flipping attack: permutes classifier output weights to simulate
training on corrupted labels.

Reference:
    Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust
    Federated Learning" (USENIX Security 2020).
"""

from typing import Dict, List

import torch

from dfl.attacks.base import BaseByzantineAttacker
from dfl.utils import average_state_dicts


class LabelFlipAttacker(BaseByzantineAttacker):
    """Simulates a label-flip data-poisoning attack at the model level.

    Produces an update by averaging the honest neighbours' models and
    then reversing the row order of the final classifier layer's weights
    and biases.  This is equivalent to the effect of training with fully
    flipped labels (y -> C-1-y) and is harder to detect via simple
    distance-based filters because the update magnitude stays close to
    the honest mean.
    """

    def craft_malicious_update(
        self,
        honest_updates: List[Dict[str, torch.Tensor]],
        node_id: int,
    ) -> Dict[str, torch.Tensor]:
        if not honest_updates:
            return {}

        avg = average_state_dicts(honest_updates)
        malicious = dict(avg)

        # Find the last weight and bias tensors (classifier head)
        weight_key = None
        bias_key = None
        for key in reversed(list(avg.keys())):
            if "weight" in key and avg[key].dim() == 2 and weight_key is None:
                weight_key = key
            elif "bias" in key and avg[key].dim() == 1 and bias_key is None:
                bias_key = key
            if weight_key and bias_key:
                break

        # Flip: reverse the class ordering in the classifier
        if weight_key is not None:
            malicious[weight_key] = torch.flip(avg[weight_key], [0])
        if bias_key is not None:
            malicious[bias_key] = torch.flip(avg[bias_key], [0])

        return malicious
