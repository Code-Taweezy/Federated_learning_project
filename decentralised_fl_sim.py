"""Entry point for the decentralised FL simulator.

This file re-exports public classes from the dfl package so that existing
import statements such as 'from decentralised_fl_sim import NetworkGraph'
continue to work. New code should import directly from the dfl package.
"""

from dfl.config import SimulationConfig
from dfl.network import NetworkGraph
from dfl.attacks.base import BaseByzantineAttacker as ByzantineAttacker
from dfl.attacks.directed import DirectedAttacker
from dfl.attacks.gaussian import GaussianAttacker
from dfl.aggregators.fedavg import FedAvgAggregator
from dfl.aggregators.balance import BALANCEAggregator
from dfl.aggregators.ubar import UBARAggregator
from dfl.node import FederatedNode
from dfl.simulator import DecentralisedSimulator
from dfl.utils import average_state_dicts as _average_state_dicts
from dfl.cli import main

if __name__ == "__main__":
    main()
