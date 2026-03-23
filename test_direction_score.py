"""Quick test of the direction score logic with peer median reference."""

import torch
import numpy as np
from dfl.verification.layer import VerificationLayer

# Test the direction score computation logic manually
def test_direction_scores():
    """Test that peer median reference correctly separates attackers from honest."""

    # Simulate 7 neighbors: 5 honest + 2 attackers
    # Honest models cluster around base model
    # Attackers send: malicious = avg - strength * sign(avg)

    torch.manual_seed(42)

    # Base model (represents honest consensus)
    base_model = {"fc.weight": torch.randn(10, 10)}

    # Honest models: small random perturbations from base
    honest_models = []
    for i in range(5):
        noise = 0.1 * torch.randn_like(base_model["fc.weight"])
        honest_models.append({"fc.weight": base_model["fc.weight"] + noise})

    # Attack models: directed attack = avg - strength * sign(avg)
    # Real attack uses strength * sign(avg) where strength is configurable
    # Default attack_strength in config is typically 0.5-1.0
    # This creates large anti-correlated deviations
    avg = torch.stack([m["fc.weight"] for m in honest_models]).mean(dim=0)
    attack_strength = 3.0  # Stronger attack to simulate real scenarios
    attack_model = {"fc.weight": avg - attack_strength * torch.sign(avg)}
    attack_models = [attack_model, {"fc.weight": avg - attack_strength * torch.sign(avg)}]

    # Combine all models
    all_models = honest_models + attack_models

    # Simulate neighbor_models dict
    node_idx = 0
    neighbor_models = {}
    all_senders = list(range(7))
    for i, model in enumerate(all_models):
        neighbor_models[(node_idx, i)] = model

    print("=== Testing Direction Score with Peer Median Reference ===\n")

    # Test each sender
    for sender_id in all_senders:
        is_attacker = sender_id >= 5

        # Manually compute what the function does
        sender_model = neighbor_models[(node_idx, sender_id)]

        # Peer median excluding sender
        peer_models = [
            neighbor_models[(node_idx, s)]
            for s in all_senders if s != sender_id
        ]

        # Compute peer median
        keys = list(sender_model.keys())
        stacked = torch.stack([m["fc.weight"].float() for m in peer_models])
        ref = stacked.median(dim=0).values

        ref_vec = ref.flatten()
        sender_vec = sender_model["fc.weight"].flatten()

        # Deviation
        deviation = sender_vec - ref_vec

        # Magnitude
        dev_norm = torch.norm(deviation).item()
        ref_norm = torch.norm(ref_vec).item()
        rel_mag = dev_norm / max(ref_norm, 1e-9)

        # Direction (cosine)
        dot = torch.sum(deviation * ref_vec).item()
        cosine = dot / (dev_norm * ref_norm) if dev_norm > 1e-9 else 0.0

        # Score logic
        mag_threshold = 2.5
        dir_threshold = -0.5
        is_large = rel_mag > mag_threshold
        is_anti = cosine < dir_threshold

        # New adaptive scoring
        mag_risk = max(0.0, rel_mag - 0.5) / 2.0
        dir_risk = max(0.0, -cosine)
        combined_risk = mag_risk * dir_risk
        score = max(0.0, min(1.0, 1.0 - combined_risk * 2.0))

        label = "ATK" if is_attacker else "honest"
        print(f"Sender {sender_id} ({label:6s}): rel_mag={rel_mag:.3f}, cosine={cosine:.3f}, "
              f"large={is_large}, anti={is_anti}, score={score:.3f}")

    print("\n=== Analysis ===")
    honest_scores = []
    attack_scores = []

    for sender_id in all_senders:
        is_attacker = sender_id >= 5
        sender_model = neighbor_models[(node_idx, sender_id)]
        peer_models = [neighbor_models[(node_idx, s)] for s in all_senders if s != sender_id]
        stacked = torch.stack([m["fc.weight"].float() for m in peer_models])
        ref = stacked.median(dim=0).values

        ref_vec = ref.flatten()
        sender_vec = sender_model["fc.weight"].flatten()
        deviation = sender_vec - ref_vec

        dev_norm = torch.norm(deviation).item()
        ref_norm = torch.norm(ref_vec).item()
        rel_mag = dev_norm / max(ref_norm, 1e-9)

        dot = torch.sum(deviation * ref_vec).item()
        cosine = dot / (dev_norm * ref_norm) if dev_norm > 1e-9 else 0.0

        mag_threshold = 2.5
        dir_threshold = -0.5
        is_large = rel_mag > mag_threshold
        is_anti = cosine < dir_threshold

        # New adaptive scoring
        mag_risk = max(0.0, rel_mag - 0.5) / 2.0
        dir_risk = max(0.0, -cosine)
        combined_risk = mag_risk * dir_risk
        score = max(0.0, min(1.0, 1.0 - combined_risk * 2.0))

        if is_attacker:
            attack_scores.append(score)
        else:
            honest_scores.append(score)

    print(f"Honest scores: min={min(honest_scores):.3f}, max={max(honest_scores):.3f}, mean={np.mean(honest_scores):.3f}")
    print(f"Attack scores: min={min(attack_scores):.3f}, max={max(attack_scores):.3f}, mean={np.mean(attack_scores):.3f}")

    threshold = 0.30
    honest_below = sum(1 for s in honest_scores if s < threshold)
    attack_below = sum(1 for s in attack_scores if s < threshold)
    print(f"\nWith threshold {threshold}:")
    print(f"  Honest flagged (false positives): {honest_below}/{len(honest_scores)}")
    print(f"  Attackers flagged (true positives): {attack_below}/{len(attack_scores)}")

    if honest_below == 0 and attack_below == len(attack_scores):
        print("\nPERFECT SEPARATION!")
    else:
        print("\nSeparation not perfect - needs tuning")

if __name__ == "__main__":
    test_direction_scores()
