"""Debug script to diagnose accuracy collapse.

Run with: python debug_collapse.py
"""

import torch
import numpy as np
from dfl.config import SimulationConfig
from dfl.simulator import DecentralisedSimulator

def check_model_health(states, label):
    """Check for NaN/Inf and print weight statistics."""
    print(f"\n{label}:")
    all_weights = []
    for i, state in enumerate(states):
        has_nan = any(torch.isnan(v).any().item() for v in state.values())
        has_inf = any(torch.isinf(v).any().item() for v in state.values())
        norm = sum(torch.sum(v.float()**2).item() for v in state.values())**0.5
        all_weights.append(norm)
        if has_nan or has_inf:
            print(f"  Node {i}: NaN={has_nan}, Inf={has_inf}, norm={norm:.2f}")
    print(f"  Norms: min={min(all_weights):.2f}, max={max(all_weights):.2f}, mean={np.mean(all_weights):.2f}")

def run_debug():
    # Test 1: No attacks (should show accuracy increasing)
    print("="*60)
    print("TEST 1: NO ATTACKS (baseline)")
    print("="*60)

    config = SimulationConfig(
        dataset='femnist',  # FEMNIST dataset now available
        num_nodes=10,
        num_rounds=5,
        aggregation='balance',
        attack_type='directed',
        attack_ratio=0.0,  # NO ATTACKS
        attack_strength=0.5,
        verification_enabled=False,
        k_neighbors=4,
        local_epochs=1,
        seed=42,
    )

    sim = DecentralisedSimulator(config)

    # Check initial model health
    initial_states = [n.get_model_state() for n in sim.nodes]
    check_model_health(initial_states, "Initial states")

    results = sim.run()

    print("\nAccuracies per round:")
    for r, accs in enumerate(results['accuracies']):
        print(f"  Round {r}: avg={np.mean(accs):.4f}")

    # Test 2: With attacks
    print("\n" + "="*60)
    print("TEST 2: WITH ATTACKS (30%)")
    print("="*60)

    config2 = SimulationConfig(
        dataset='shakespeare',  # Using shakespeare since femnist data is missing
        num_nodes=10,
        num_rounds=5,
        aggregation='balance',
        attack_type='directed',
        attack_ratio=0.3,  # 30% attacks
        attack_strength=0.5,
        verification_enabled=False,
        k_neighbors=4,
        local_epochs=1,
        seed=42,
    )

    sim2 = DecentralisedSimulator(config2)

    # Hook into aggregation to debug
    original_aggregate = sim2.nodes[0].aggregator.aggregate

    def debug_aggregate(own_model, neighbor_models, neighbor_indices=None):
        # Print distances
        from dfl.utils import model_distance
        threshold = sim2.nodes[0].aggregator._compute_threshold(own_model)
        print(f"\n  [Debug] Threshold: {threshold:.2f}")
        for i, nm in enumerate(neighbor_models):
            dist = model_distance(own_model, nm)
            idx = neighbor_indices[i] if neighbor_indices else i
            status = "ACCEPT" if dist <= threshold else "REJECT"
            print(f"    Neighbor {idx}: dist={dist:.2f} -> {status}")

        return original_aggregate(own_model, neighbor_models, neighbor_indices)

    # Apply debug hook to first node only (to reduce output)
    sim2.nodes[0].aggregator.aggregate = debug_aggregate

    results2 = sim2.run()

    print("\nAccuracies per round:")
    for r, accs in enumerate(results2['accuracies']):
        print(f"  Round {r}: avg={np.mean(accs):.4f}")

    # Check model health after round 1
    final_states = [n.get_model_state() for n in sim2.nodes]
    check_model_health(final_states, "Final states (after 5 rounds)")

if __name__ == "__main__":
    run_debug()
