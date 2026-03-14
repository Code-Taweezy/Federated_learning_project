"""Aggregator registry and factory function."""

from dfl.aggregators.fedavg import FedAvgAggregator
from dfl.aggregators.balance import BALANCEAggregator
from dfl.aggregators.ubar import UBARAggregator
from dfl.aggregators.krum import KrumAggregator, MultiKrumAggregator
from dfl.aggregators.trimmed import TrimmedMeanAggregator, CoordinateMedianAggregator

# Registry mapping aggregation names to their implementing classes
AGGREGATOR_REGISTRY = {
    "fedavg": FedAvgAggregator,
    "balance": BALANCEAggregator,
    "ubar": UBARAggregator,
    "krum": KrumAggregator,
    "multikrum": MultiKrumAggregator,
    "trimmed_mean": TrimmedMeanAggregator,
    "median": CoordinateMedianAggregator,
}


def register_aggregator(name: str, cls) -> None:
    """Register a custom aggregator class at runtime.

    This allows users to add new aggregation strategies without modifying
    the package source code.
    """
    AGGREGATOR_REGISTRY[name] = cls


def create_aggregator(name: str, node_id: int, config, **kwargs):
    """Create the appropriate aggregator instance based on name.

    Extra keyword arguments are forwarded to the aggregator constructor.
    """
    cls = AGGREGATOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown aggregation '{name}'. "
            f"Available: {list(AGGREGATOR_REGISTRY.keys())}"
        )
    return cls(node_id=node_id, config=config, **kwargs)
