"""Attack strategy registry and factory function."""

from dfl.attacks.directed import DirectedAttacker
from dfl.attacks.gaussian import GaussianAttacker
from dfl.attacks.label_flip import LabelFlipAttacker
from dfl.attacks.alie import ALIEAttacker

# Registry mapping attack type names to their implementing classes
ATTACK_REGISTRY = {
    "directed": DirectedAttacker,
    "gaussian": GaussianAttacker,
    "label_flip": LabelFlipAttacker,
    "alie": ALIEAttacker,
}


def register_attack(name: str, cls) -> None:
    """Register a custom attack class at runtime.

    This allows users to add new attack strategies without modifying
    the package source code.
    """
    ATTACK_REGISTRY[name] = cls


def create_attacker(config):
    """Create the appropriate attacker instance based on config.attack_type.

    Returns None when attack_ratio is zero (no attack configured).
    """
    if config.attack_ratio <= 0:
        return None

    cls = ATTACK_REGISTRY.get(config.attack_type)
    if cls is None:
        raise ValueError(
            f"Unknown attack type '{config.attack_type}'. "
            f"Available: {list(ATTACK_REGISTRY.keys())}"
        )
    return cls(config)
