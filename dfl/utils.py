"""Shared utility functions used across the dfl package."""

import random
from typing import Dict, List

import numpy as np
import torch


def average_state_dicts(
    models: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Average a list of model state dicts parameter-wise."""
    if not models:
        return {}
    avg = {}
    for key in models[0].keys():
        stacked = torch.stack([m[key].float() for m in models])
        avg[key] = stacked.mean(dim=0).to(models[0][key].dtype)
    return avg


def model_distance(
    model1: Dict[str, torch.Tensor], model2: Dict[str, torch.Tensor]
) -> float:
    """Compute the L2 distance between two model state dicts."""
    total = 0.0
    for key in model1.keys():
        diff = model1[key].float() - model2[key].float()
        total += torch.sum(diff * diff).item()
    return float(np.sqrt(total))


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
