# Threat Model

This document formally defines the adversary model, attack surface, trust
assumptions, and defense scope for the decentralised federated learning (DFL)
simulator.

## 1. System Model

| Property | Setting |
|---|---|
| **Architecture** | Fully decentralised (no central server) |
| **Communication** | Synchronous rounds; each node exchanges model state dicts with its graph neighbours |
| **Topologies** | Ring, Fully-connected, K-regular |
| **Data** | Each node holds a private, non-IID partition (Dirichlet-alpha controlled) |
| **Aggregation** | Per-node neighbourhood aggregation using a pluggable strategy |

## 2. Adversary Capabilities

We consider a **Byzantine omniscient** adversary that:

- **Controls** up to `f` of `n` nodes (`attack_ratio <= 0.5`, i.e. `f <= n/2`).
- **Knows** the honest model updates received by compromised nodes (white-box
  access to neighbour states each round).
- **Can craft** arbitrary model state dicts to replace a compromised node's
  outgoing message.
- **Cannot** modify network topology, intercept or alter honest-to-honest
  messages, break cryptographic primitives, or compromise additional nodes
  after initialisation.

### What the adversary *cannot* do

| Excluded capability | Reason |
|---|---|
| Adaptive node compromise at runtime | Node sets are fixed at initialisation to model a static Sybil threat |
| Network-layer attacks (DoS, partitioning) | Out of scope; we assume a reliable transport layer |
| Poisoning the test/validation data | Nodes hold private, immutable evaluation sets |
| Colluding across non-adjacent nodes | Communication follows the topology graph; cross-graph collusion is equivalent to a higher `f` and is captured by varying `attack_ratio` |

## 3. Attack Strategies

| Attack | Key | Description | Threat Level |
|---|---|---|---|
| **Directed deviation** | `directed` | Computes honest average, then deviates in the opposite direction scaled by `attack_strength` | High — directly maximises damage to convergence |
| **Gaussian noise** | `gaussian` | Replaces model parameters with scaled Gaussian noise | Medium — easily detected by distance-based defences |
| **Label flipping** | `label_flip` | Reverses the classifier head weight rows, simulating a systematic label permutation at the model level | Medium — subtle effect concentrated in the final layer |
| **A Little Is Enough (ALIE)** | `alie` | Shifts parameters by `z_max` standard deviations below the honest mean, staying close enough to evade distance-based filters | High — specifically designed to bypass robust aggregators |

## 4. Defense Layers

### 4.1 Pre-acceptance (Aggregation)

Each node applies a configurable aggregation strategy before accepting
neighbour updates:

| Aggregator | Key | Byzantine resilience |
|---|---|---|
| FedAvg | `fedavg` | None — baseline, accepts all |
| BALANCE | `balance` | Distance-threshold filtering with weighted blending |
| UBAR | `ubar` | Two-stage outlier removal |
| Krum / Multi-Krum | `krum` / `multikrum` | Selects the update(s) closest to other updates |
| Trimmed Mean | `trimmed_mean` | Coordinate-wise trimming of extreme parameter values |
| Coordinate Median | `median` | Coordinate-wise median, breakdown point of 50% |

### 4.2 Post-acceptance (Verification Layer)

A four-phase verification mechanism runs *after* aggregation each round to
catch attacks that slipped past the first layer:

| Phase | Purpose | Key signals |
|---|---|---|
| **Phase 1 — Flag** | Remove accepted neighbours that are *persistently* suspicious | Low trust score, high drift/peer-dev history, message distance, performance delta (needs `>= phase1_min_signals` signals for `phase1_consecutive_required` consecutive rounds) |
| **Phase 2 — Rescue** | Re-admit rejected neighbours whose messages look benign | Trust >= threshold, peer-dev <= median, drift within sigma band, z-score < z_high, no accuracy loss (needs `>= phase2_min_signals` signals) |
| **Phase 3 — Re-aggregate** | Rebuild the model if Phase 1 or 2 changed the accepted set | Blending formula: `alpha * own + (1-alpha) * mean(accepted)` |
| **Phase 4 — Trust decay** | Update trust scores for nodes not touched in Phases 1-2 | Exponential decay toward neutral, penalise flagged, boost rescued |

### Trust Score Dynamics

- **Initial trust:** 0.5 (neutral)
- **Penalty per flag:** -0.2 (one flag drops trust to 0.3, below Phase 1 threshold of 0.35)
- **Boost per rescue:** +0.05 (4 successful rescues needed to recover from one penalty)
- **Decay factor:** 0.95 per round (half-life ~14 rounds)

## 5. Security Properties

### What this system **does** defend against

1. **Model poisoning** attacks where compromised nodes send manipulated
   gradient or model updates.
2. **Free-riding** nodes that send noise instead of genuine training updates.
3. **Stealthy attacks** (e.g. ALIE) via the post-acceptance verification
   layer's multi-signal detection.

### What this system **does not** defend against

1. **Data poisoning** — compromised nodes that poison their local training
   data before computing genuine updates. The simulator assumes honest
   training on the local dataset.
2. **Backdoor/trojan attacks** — targeted mis-classification on trigger
   patterns embedded in the training data.
3. **Privacy attacks** — gradient inversion or membership inference. No
   differential privacy or secure aggregation is applied.
4. **Network-layer attacks** — denial of service, eclipse, or Sybil attacks
   beyond the initial fixed compromise.

## 6. Evaluation Criteria

Defenses are evaluated along four axes:

| Metric | Description |
|---|---|
| **Accuracy** | Global and per-class accuracy on honest vs compromised nodes |
| **Detection quality** | Precision, Recall, F1, and Attack Success Rate (ASR) per round |
| **Robustness** | Accuracy degradation relative to the no-attack (`attack_ratio=0`) baseline |
| **Overhead** | Wall-clock time per round (with/without verification) and communication bytes |

All metrics are reported with 95% confidence intervals across multiple seeds.

## 7. Assumptions

1. Nodes run identical model architectures and share a common initialisation.
2. The topology graph is known to all nodes (public structure).
3. Local test/validation sets are representative of the node's true
   distribution and are not accessible to the adversary.
4. `attack_ratio <= 0.5` (honest majority).
5. Communication is synchronous — all nodes complete their round before the
   next round begins.
