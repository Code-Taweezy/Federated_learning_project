>**Research Question:** How can post-acceptance attacks be efficiently detected in decentralised federated learning? 

**This project is under active development as part of an ongoing research project** 

## Overview 

This repository contains the experimental infrastructure for evaluating decentralised federated learning (DFL) defenses under adversarial (Byzantine) conditions. The simulator models honest and malicious client interactions across configurable network topologies, attack strategies, defense mechanisms, and datasets — enabling systematic, reproducible evaluation of robustness, detection effectiveness, and computational efficiency.

The purpose of this simulation is to establish baseline performance data for existing DFL defense mechanisms, which will later be compared against a novel lightweight post-acceptance verification layer designed to detect attacks that bypass initial filtering.

## Features

### Configurable DFL Simulator 

- **Datasets:** FEMNIST and Shakespeare (via [LEAF]https://github.com/TalwalkarLab/leaf) benchmarks

- **Defense / Aggregation Mechanisms:**
  - `FedAvg` — Baseline federated averaging
  - `BALANCE` — Distance-threshold filtering with weighted aggregation
  - `UBAR` — Two-stage Byzantine-robust aggregation
  - `SketchGuard` — Compressed sketch-based filtering
- **Attack Types:**
  - `Directed` — Pushes models toward a target class
  - `Gaussian` — Adds noise to gradients
- **Network Topologies:** Ring, Fully Connected, K-Regular
- **Attack Ratio:** Configurable from 0.0 to 0.5 (proportion of compromised nodes)
- Supports both **CLI arguments** and an **interactive configuration mode**

### Live Dashboard

An interactive dashboard (`results_dashboard.py`) provides real-time monitoring of:
- Network topology graph (honest vs. malicious nodes)
- Accuracy over training rounds (honest vs. compromised)
- Node acceptance rates across rounds

The dashboard also allows users to modify experiment parameters between runs without manual code changes.

### Automated Data Collection

Experimental results are logged in real time to Google Sheets via the Google Cloud API (`sheets_exporter.py`), recording:
- Compute time
- Final model accuracy
- Honest nodes accepted
- Compromised nodes accepted
- Attack impact (relative to 0.0 attack baseline)

## Getting Started

### Prerequisites

- Python 3.9+
- [PyTorch](https://pytorch.org/) 2.0+
- (Optional) CUDA-compatible GPU for accelerated training
  
### Installations 
```bash
pip install -r requirements.txt
```

### Running Automated experiments 
To execute the full experiment suite: 
```bash
python run_experiments.py
```
## Next Steps

After completing baseline data collection and analysis, a **lightweight post-acceptance verification layer** will be implemented. This mechanism will target weaknesses identified in the baseline results and will be evaluated under the same experimental conditions, enabling direct comparison against existing methods across robustness, detection accuracy, and computational efficiency.

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Visualisation:** Matplotlib, Seaborn
- **Data Processing:** NumPy, Pandas, scikit-learn
- **Graph Modelling:** NetworkX
- **Data Export:** gspread, Google Auth

## License

All rights reserved. This repository is part of an active research project and is not licensed for reuse, modification, or distribution at this time. Please contact the author for any inquiries.
