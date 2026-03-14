"""Simulation configuration for decentralised federated learning experiments."""

from dataclasses import dataclass
from typing import Optional


# Valid options for configuration fields
VALID_DATASETS = ("femnist", "shakespeare", "celeba", "reddit")
VALID_TOPOLOGIES = ("ring", "fully", "k-regular")
VALID_AGGREGATIONS = ("fedavg", "balance", "ubar", "krum", "multikrum", "trimmed_mean", "median")
VALID_ATTACK_TYPES = ("directed", "gaussian", "label_flip", "alie")


@dataclass
class SimulationConfig:
    """Holds every tuneable parameter for a single simulation run."""

    dataset: str = "femnist"
    num_nodes: int = 32
    num_rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.01
    seed: int = 42

    # Graph topology configuration
    topology: str = "ring"
    k_neighbors: int = 4

    # Aggregation algorithm configuration
    aggregation: str = "fedavg"

    # Attack configuration
    attack_type: str = "directed"
    attack_ratio: float = 0.0
    attack_strength: float = 1.0

    # Algorithm-specific parameters
    balance_alpha: float = 0.5
    balance_gamma: float = 2.0
    balance_kappa: float = 1.0
    ubar_rho: float = 0.4

    # Krum parameters
    krum_multi_k: Optional[int] = None

    # Trimmed Mean parameters
    trimmed_mean_beta: float = 0.1

    # ALIE attack parameter
    alie_z_max: float = 1.0

    partition_alpha: float = 0.5

    # Verification layer parameters
    verification_enabled: bool = True
    verification_epsilon: float = 0.05
    trust_decay: float = 0.95
    trust_initial: float = 0.5
    trust_penalty: float = 0.2
    trust_boost: float = 0.05
    z_low: float = 1.5
    z_high: float = 2.5
    verification_history_window: int = 10
    rescue_revocation_rounds: int = 3

    # Verification layer thresholds (previously hard-coded)
    phase1_trust_threshold: float = 0.35
    phase1_min_signals: int = 3
    phase1_consecutive_required: int = 2
    phase2_trust_threshold: float = 0.4
    phase2_min_accuracy: float = 0.05
    phase2_min_signals: int = 4
    phase2_drift_sigma_factor: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.attack_ratio <= 0.5):
            raise ValueError(
                f"attack_ratio must be between 0.0 and 0.5, got {self.attack_ratio}"
            )
        if self.attack_type not in VALID_ATTACK_TYPES:
            raise ValueError(f"Unknown attack_type: {self.attack_type}")
        if self.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
