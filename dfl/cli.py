#Command-line interface for the simulator.

import argparse
import os

from dfl.config import SimulationConfig, VALID_ATTACK_TYPES, VALID_AGGREGATIONS
from dfl.simulator import DecentralisedSimulator


DATASETS = ["femnist", "shakespeare"]
AGGREGATORS = list(VALID_AGGREGATIONS)
TOPOLOGIES = ["ring", "fully", "k-regular"]


def main():
    #Entry point for the decentralised FL simulator."""
    parser = argparse.ArgumentParser(description="Decentralised FL Simulator")

    # Basic parameters
    parser.add_argument("--dataset", type=str, default="femnist", choices=DATASETS)
    parser.add_argument("--num-nodes", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)

    # Topology
    parser.add_argument("--topology", type=str, default="ring", choices=TOPOLOGIES)
    parser.add_argument("--k", type=int, default=4)

    # Aggregation
    parser.add_argument("--aggregation", type=str, default="fedavg", choices=AGGREGATORS)

    # Attack
    parser.add_argument("--attack-ratio", type=float, default=0.0)
    parser.add_argument(
        "--attack-type", type=str, default="directed", choices=list(VALID_ATTACK_TYPES)
    )
    parser.add_argument("--attack-strength", type=float, default=1.0)

    # Algorithm parameters
    parser.add_argument("--balance-gamma", type=float, default=2.0)
    parser.add_argument("--balance-kappa", type=float, default=1.0)
    parser.add_argument("--balance-alpha", type=float, default=0.5)
    parser.add_argument("--ubar-rho", type=float, default=0.4)

    # Krum parameters
    parser.add_argument(
        "--krum-multi-k", type=int, default=None,
        help="Number of models to select for Multi-Krum (None means single Krum)",
    )

    # Trimmed Mean parameters
    parser.add_argument(
        "--trimmed-mean-beta", type=float, default=0.1,
        help="Fraction to trim from each end in coordinate-wise trimmed mean",
    )

    # ALIE attack parameters
    parser.add_argument(
        "--alie-z-max", type=float, default=1.0,
        help="Maximum z-score for ALIE attack perturbation",
    )

    # Partitioning
    parser.add_argument(
        "--partition-alpha", type=float, default=0.5,
        help="Dirichlet alpha for non-IID data partitioning (lower = more heterogeneous)",
    )

    # Verification layer
    parser.add_argument(
        "--verification", dest="verification", action="store_true", default=True,
        help="Enable post-acceptance verification layer (default: enabled)",
    )
    parser.add_argument(
        "--no-verification", dest="verification", action="store_false",
        help="Disable post-acceptance verification layer",
    )
    parser.add_argument("--verification-epsilon", type=float, default=0.05)
    parser.add_argument("--trust-decay", type=float, default=0.9)
    parser.add_argument("--trust-initial", type=float, default=0.5)
    parser.add_argument("--trust-penalty", type=float, default=0.3)
    parser.add_argument("--trust-boost", type=float, default=0.05)
    parser.add_argument("--z-low", type=float, default=1.5)
    parser.add_argument("--z-high", type=float, default=2.5)
    parser.add_argument("--verification-history-window", type=int, default=10)
    parser.add_argument("--rescue-revocation-rounds", type=int, default=3)

    # Verification layer thresholds (sensitivity analysis)
    parser.add_argument("--phase1-trust-threshold", type=float, default=0.35,
                        help="Trust threshold below which Phase 1 flags a node as suspicious")
    parser.add_argument("--phase1-min-signals", type=int, default=2,
                        help="Minimum number of suspicious signals (out of 3) for Phase 1 flagging")
    parser.add_argument("--phase1-consecutive-required", type=int, default=1,
                        help="Consecutive suspicious rounds required before Phase 1 acts")
    parser.add_argument("--phase2-trust-threshold", type=float, default=0.4,
                        help="Minimum trust score for Phase 2 rescue eligibility")
    parser.add_argument("--phase2-min-accuracy", type=float, default=0.001,
                        help="Minimum model accuracy before Phase 2 rescue is attempted")
    parser.add_argument("--phase2-min-signals", type=int, default=4,
                        help="Minimum number of benign signals (out of 5) for Phase 2 rescue")
    parser.add_argument("--phase2-drift-sigma-factor", type=float, default=1.0,
                        help="Drift must be within mean + factor*sigma for Phase 2 rescue")

    # Seed and output
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-run, e.g. '42,123,456'")
    parser.add_argument("--output", type=str, default="results/experiment.json")

    args = parser.parse_args()

    dataset = args.dataset
    aggregation = args.aggregation
    topology = args.topology
    num_nodes = args.num_nodes
    num_rounds = args.rounds
    attack_ratio = args.attack_ratio
    output_path = args.output

    # Parse multi-seed mode
    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    all_results = []
    for seed_idx, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\nSeed run {seed_idx + 1}/{len(seeds)} (seed={seed})")
            seed_output = output_path.replace(".json", f"_seed{seed}.json")
        else:
            seed_output = output_path

        config = SimulationConfig(
            dataset=dataset,
            num_nodes=num_nodes,
            num_rounds=num_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            topology=topology,
            k_neighbors=args.k,
            aggregation=aggregation,
            attack_ratio=attack_ratio,
            attack_type=args.attack_type,
            attack_strength=args.attack_strength,
            balance_gamma=args.balance_gamma,
            balance_kappa=args.balance_kappa,
            balance_alpha=args.balance_alpha,
            ubar_rho=args.ubar_rho,
            krum_multi_k=args.krum_multi_k,
            trimmed_mean_beta=args.trimmed_mean_beta,
            alie_z_max=args.alie_z_max,
            partition_alpha=args.partition_alpha,
            seed=seed,
            verification_enabled=args.verification,
            verification_epsilon=args.verification_epsilon,
            trust_decay=args.trust_decay,
            trust_initial=args.trust_initial,
            trust_penalty=args.trust_penalty,
            trust_boost=args.trust_boost,
            z_low=args.z_low,
            z_high=args.z_high,
            verification_history_window=args.verification_history_window,
            rescue_revocation_rounds=args.rescue_revocation_rounds,
            phase1_trust_threshold=args.phase1_trust_threshold,
            phase1_min_signals=args.phase1_min_signals,
            phase1_consecutive_required=args.phase1_consecutive_required,
            phase2_trust_threshold=args.phase2_trust_threshold,
            phase2_min_accuracy=args.phase2_min_accuracy,
            phase2_min_signals=args.phase2_min_signals,
            phase2_drift_sigma_factor=args.phase2_drift_sigma_factor,
        )

        simulator = DecentralisedSimulator(config)
        results = simulator.run()
        all_results.append(results)

        out_dir = os.path.dirname(seed_output) or "."
        os.makedirs(out_dir, exist_ok=True)
        simulator.save_results(seed_output)

    # Multi-seed aggregation
    if len(seeds) > 1:
        _aggregate_multi_seed(seeds, output_path, all_results)


def _compute_ci(values, confidence=0.95):
    #Compute mean, std, and confidence interval for a list of values.
    import numpy as np
    from scipy import stats

    values = [v for v in values if v is not None]
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    mu = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    if len(values) > 1 and sd > 0:
        ci = stats.t.interval(
            confidence, df=len(values) - 1, loc=mu,
            scale=sd / np.sqrt(len(values)),
        )
        return {"mean": mu, "std": sd, "ci_lower": float(ci[0]), "ci_upper": float(ci[1])}
    return {"mean": mu, "std": sd, "ci_lower": mu, "ci_upper": mu}


def _aggregate_multi_seed(seeds, output_path, all_results):
    """Aggregate results from multiple seed runs with 95% CIs.

    Aggregates accuracy, loss, detection metrics (precision, recall, F1,
    ASR), drift, consensus, and overhead across seeds.
    """
    import json
    import numpy as np

    num_rounds = len(all_results[0]["accuracies"])

    per_round_agg = []
    for r in range(num_rounds):
        round_accs = [float(np.mean(res["accuracies"][r])) for res in all_results]
        round_losses = [float(np.mean(res["losses"][r])) for res in all_results]

        entry = {
            "round": r,
            "accuracy": _compute_ci(round_accs),
            "loss": _compute_ci(round_losses),
        }

        # Honest / compromised accuracy
        honest_accs = [
            res["honest_accuracies"][r]
            for res in all_results
            if r < len(res.get("honest_accuracies", []))
        ]
        if honest_accs:
            entry["honest_accuracy"] = _compute_ci(honest_accs)

        comp_accs = [
            res["compromised_accuracies"][r]
            for res in all_results
            if r < len(res.get("compromised_accuracies", []))
        ]
        if comp_accs:
            entry["compromised_accuracy"] = _compute_ci(comp_accs)

        # Drift
        drifts = [
            res["drift_per_round"][r]["mean"]
            for res in all_results
            if r < len(res.get("drift_per_round", []))
        ]
        if drifts:
            entry["drift_mean"] = _compute_ci(drifts)

        # Consensus
        consensus = [
            res["consensus_score_per_round"][r]
            for res in all_results
            if r < len(res.get("consensus_score_per_round", []))
        ]
        if consensus:
            entry["consensus"] = _compute_ci(consensus)

        # Overhead
        time_wo = [
            res["overhead_time"]["without_detection"][r]
            for res in all_results
            if r < len(res.get("overhead_time", {}).get("without_detection", []))
        ]
        if time_wo:
            entry["time_without_detection"] = _compute_ci(time_wo)

        time_w = [
            res["overhead_time"]["with_detection"][r]
            for res in all_results
            if r < len(res.get("overhead_time", {}).get("with_detection", []))
        ]
        if time_w:
            entry["time_with_detection"] = _compute_ci(time_w)

        # Communication overhead (bytes)
        comm = [
            res["communication_overhead"][r]
            for res in all_results
            if r < len(res.get("communication_overhead", []))
        ]
        if comm:
            entry["communication_bytes"] = _compute_ci(comm)

        per_round_agg.append(entry)

    # Final summary
    final_accs = [float(np.mean(res["accuracies"][-1])) for res in all_results]
    final_losses = [float(np.mean(res["losses"][-1])) for res in all_results]

    summary_agg = {
        "final_accuracy": _compute_ci(final_accs),
        "final_loss": _compute_ci(final_losses),
    }

    # Detection metrics summary
    for metric_key in ("precision", "recall", "f1", "asr"):
        vals = []
        for res in all_results:
            det_rounds = res.get("detection_metrics_per_round", [])
            if det_rounds:
                vals.append(det_rounds[-1].get(metric_key, 0.0))
        if vals:
            summary_agg[f"final_{metric_key}"] = _compute_ci(vals)

    agg_output = {
        "seeds_used": seeds,
        "num_seeds": len(seeds),
        "confidence_level": 0.95,
        "per_round": per_round_agg,
        "summary": summary_agg,
    }

    agg_path = output_path.replace(".json", "_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(agg_output, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")


if __name__ == "__main__":
    main()
