"""Decentralised Federated Learning simulator package.

This package provides a modular framework for simulating decentralised
federated learning under Byzantine attacks, with pluggable aggregation
algorithms, attack strategies, and a post-acceptance verification layer.

Public API is re-exported here for convenience.
"""

from dfl.config import SimulationConfig
from dfl.network import NetworkGraph
from dfl.attacks.base import BaseByzantineAttacker
from dfl.attacks import create_attacker
from dfl.aggregators.fedavg import FedAvgAggregator
from dfl.aggregators.balance import BALANCEAggregator
from dfl.aggregators.ubar import UBARAggregator
from dfl.aggregators.krum import KrumAggregator, MultiKrumAggregator
from dfl.aggregators.trimmed import TrimmedMeanAggregator, CoordinateMedianAggregator
from dfl.aggregators import create_aggregator
from dfl.attacks.label_flip import LabelFlipAttacker
from dfl.attacks.alie import ALIEAttacker
from dfl.node import FederatedNode
from dfl.simulator import DecentralisedSimulator
from dfl.metrics import MetricsTracker
from dfl.cli import main

__all__ = [
    "SimulationConfig",
    "NetworkGraph",
    "BaseByzantineAttacker",
    "create_attacker",
    "LabelFlipAttacker",
    "ALIEAttacker",
    "FedAvgAggregator",
    "BALANCEAggregator",
    "UBARAggregator",
    "KrumAggregator",
    "MultiKrumAggregator",
    "TrimmedMeanAggregator",
    "CoordinateMedianAggregator",
    "create_aggregator",
    "FederatedNode",
    "DecentralisedSimulator",
    "MetricsTracker",
    "main",
]
