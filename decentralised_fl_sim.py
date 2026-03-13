import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Subset 
import numpy as np 
import random 
import time 
import json 
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass 
import argparse

from leaf_datasets import load_leaf_dataset, create_leaf_client_partitions
# Configuration Classes

@dataclass
class SimulationConfig: 

    dataset: str = "femnist"
    num_nodes: int = 32 
    num_rounds: int = 50 
    local_epochs: int = 1 
    batch_size: int = 128 
    learning_rate: float = 0.01 
    seed: int = 42 

    # Graph topology configuration
    topology: str = "ring"  # Options: ring, fully, k-regular
    k_neighbors: int = 4     # Number of neighbors for k-regular topology

    # Aggregation algorithm configuration
    aggregation: str = "fedavg"  # Options: fedavg, balance, ubar

    # Attack configuration
    attack_type: str = "directed"  # Options: directed, gaussian
    attack_ratio: float = 0.0      # Proportion of compromised nodes (0.0 to 0.5)
    attack_strength: float = 1.0   # Attack strength for directed and gaussian attacks

    # Algorithm-specific parameters
    balance_alpha: float = 0.5     # Alpha for balance aggregation
    balance_gamma: float = 2.0     # Gamma for balance aggregation
    balance_kappa: float = 1.0     # Kappa for balance aggregation
    ubar_rho: float = 0.4          # Rho for UBAR aggregation

    partition_alpha: float = 0.5   # Dirichlet alpha for non-IID partitioning (lower = more heterogeneous)

    # Verification layer parameters
    verification_enabled: bool = True          # Enable/disable verification layer
    verification_epsilon: float = 0.05         # Minimum validation improvement required before flagging/removing a peer
    trust_decay: float = 0.95                  # Exponential decay for trust score updates
    trust_initial: float = 0.5                 # Initial trust score for all nodes
    trust_penalty: float = 0.2                 # Trust reduction after a confirmed verification flag
    trust_boost: float = 0.05                  # Trust increase after a successful rescue
    z_low: float = 1.5                         # Lower bound for ambiguous z-score range
    z_high: float = 2.5                        # Upper bound for ambiguous z-score range
    verification_history_window: int = 10      # Rounds of history used by verification
    rescue_revocation_rounds: int = 3          # Reserved for backwards-compatible CLI support

    def __post_init__(self):
        if not (0.0 <= self.attack_ratio <= 0.5):
            raise ValueError(f"attack_ratio must be between 0.0 and 0.5, got {self.attack_ratio}")
        if self.attack_type not in ("directed", "gaussian"):
            raise ValueError(f"Unknown attack_type: {self.attack_type}")
        if self.aggregation not in ("fedavg", "balance", "ubar"):
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

# Topologies and Network Graph

class NetworkGraph:
    # Handles network topology for decentralised learning

    def __init__(self, num_nodes: int, topology: str, k: int = 4) :
        self.num_nodes = num_nodes
        self.topology = topology
        self.k = k
        self.adjacency_list = self.build_topology()
    
    def build_topology(self) -> Dict[int, List[int]]:
        #Builds a network topology 
        adj = {i: [] for i in range(self.num_nodes)}

        if self.topology == "ring":
            #each node connects to next and previous node
            for i in range(self.num_nodes):
                adj[i].append((i - 1) % self.num_nodes) # previous node
                adj[i].append((i + 1) % self.num_nodes) # next node
        elif self.topology == "fully":
            #each node connects to every other node
            for i in range(self.num_nodes):
                adj[i] = [j for j in range(self.num_nodes) if j != i]
        
        elif self.topology == "k-regular":
            #each node connects to k nearest neighbors (circular)
            if self.k >= self.num_nodes:
                self.k = self.num_nodes - 1 # max neighbors is num_nodes - 1
            half_k = self.k // 2

            for i in range(self.num_nodes):
                neighbors = []
                for offset in range(1, half_k + 1):
                    neighbors.append((i + offset) % self.num_nodes)
                    neighbors.append((i - offset) % self.num_nodes)
                adj[i] = list(set(neighbors))[:self.k]
        return adj 

    def get_neighbors(self, node_id: int) -> List[int]:
        #Get the neighbors of a node
        return self.adjacency_list[node_id]
    
    def get_edge_list(self) -> List[Tuple[int, int]]:
        #Get the list of edges
        edges=set()
        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors: 
                edge = tuple(sorted([node, neighbor]))
                edges.add(edge)
        return list(edges)
    
#Byzantine Attacks

def _average_state_dicts(models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average a list of model state dicts parameter-wise."""
    if not models:
        return {}
    avg = {}
    for key in models[0].keys():
        stacked = torch.stack([m[key].float() for m in models])
        avg[key] = stacked.mean(dim=0).to(models[0][key].dtype)
    return avg

class ByzantineAttacker:
    # Implements Byzantine attacks for federated learning (directed and gaussian)

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_compromised = int(config.num_nodes * config.attack_ratio)

        #Randomly selecting compromised nodes
        random.seed(config.seed)
        self.compromised_nodes = set(random.sample(range (config.num_nodes), self.num_compromised))
        print(f"Compromised {len(self.compromised_nodes)} nodes: {sorted(self.compromised_nodes)}" )
    def is_compromised(self, node_id: int) -> bool:
        #Checks if a node is compromised
        return node_id in self.compromised_nodes 
    
    def craft_malicious_update(self, honest_updates: List[Dict[str,torch.Tensor]], node_id:int ) -> Dict[str,torch.Tensor]:
        #creates a malicious model update 

        if not honest_updates: 
            # No honest updates available — return empty (zero) update
            return {}
        #compute the average of honest updates 
        avg_update = _average_state_dicts(honest_updates)

        if self.config.attack_type == "gaussian":
            # Gaussian attack: sample Gaussian noise and scale by attack_strength * sqrt(200)
            malicious = {}

            for key, param in avg_update.items():
                noise = torch.randn_like(param.float()) * self.config.attack_strength * np.sqrt(200.0)
                malicious[key] = noise.to(param.dtype)
            return malicious
        elif self.config.attack_type == "directed":
            #Directed deviation: move in opposite direction 
            malicious = {}
            for key, param in avg_update.items():
                #Estimate direction (simple: use sign of average)
                direction = torch.sign(param)

                #deviate in opposite direction 
                malicious[key] = param - self.config.attack_strength * direction
            return malicious
        else:
            raise ValueError(f"Unknown attack type: {self.config.attack_type}")
    
#Aggregation Algorithms 

class FedAvgAggregator : 
    """Standard Federated Averaging: accepts all neighbors equally."""

    def __init__ (self,node_id: int ):
        self.node_id = node_id
        self.stats = {"accepted": 0, "total": 0}
    
    def aggregate (self, own_model: Dict[str, torch.Tensor], neighbor_models: List[Dict[str, torch.Tensor]], neighbor_indices: List[int] = None) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        #simple averaging of own model and neighbor models
        all_models = [own_model] + neighbor_models 
        self.stats["total"] += len(neighbor_models)
        self.stats["accepted"] += len(neighbor_models)

        accepted = list(neighbor_indices) if neighbor_indices is not None else []
        return _average_state_dicts(all_models), accepted, []
    
class BALANCEAggregator:
    """BALANCE: Distance-threshold filtering with weighted aggregation."""

    def __init__(self, node_id: int, config: SimulationConfig, total_rounds: int):
        self.node_id = node_id
        self.config = config
        self.total_rounds = total_rounds
        self.current_round = 0
        self.stats = {"accepted": 0, "total": 0, "acceptance_rates": [], "thresholds": []}

    def _model_distance(self, model1: Dict[str, torch.Tensor], model2: Dict[str, torch.Tensor]) -> float:
        total = 0.0
        for key in model1.keys():
            diff = model1[key].float() - model2[key].float()
            total += torch.sum(diff * diff).item()
        return float(np.sqrt(total))

    def _compute_threshold(self, own_model: Dict[str, torch.Tensor]) -> float:
        own_norm_sq = 0.0
        for param in own_model.values():
            p = param.float()
            own_norm_sq += torch.sum(p * p).item()
        own_norm = float(np.sqrt(own_norm_sq))
        lambda_t = self.current_round / max(1, self.total_rounds)
        return self.config.balance_gamma * np.exp(-self.config.balance_kappa * lambda_t) * own_norm

    def aggregate(self, own_model: Dict[str, torch.Tensor], neighbor_models: List[Dict[str, torch.Tensor]], neighbor_indices: List[int] = None) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        self.current_round += 1
        if not neighbor_models:
            return own_model, [], []

        threshold = self._compute_threshold(own_model)
        self.stats["thresholds"].append(threshold)

        accepted_models = []
        accepted_indices = []
        rejected_indices = []
        for idx, neighbor_model in enumerate(neighbor_models):
            distance = self._model_distance(own_model, neighbor_model)
            neighbor_id = neighbor_indices[idx] if neighbor_indices is not None else idx
            if distance <= threshold:
                accepted_models.append(neighbor_model)
                accepted_indices.append(neighbor_id)
            else:
                rejected_indices.append(neighbor_id)

        if not accepted_models:
            # Fallback: accept closest
            closest_idx = min(range(len(neighbor_models)),
                              key=lambda i: self._model_distance(own_model, neighbor_models[i]))
            closest_id = neighbor_indices[closest_idx] if neighbor_indices is not None else closest_idx
            accepted_models.append(neighbor_models[closest_idx])
            accepted_indices.append(closest_id)
            if closest_id in rejected_indices:
                rejected_indices.remove(closest_id)

        self.stats["total"] += len(neighbor_models)
        self.stats["accepted"] += len(accepted_models)
        self.stats["acceptance_rates"].append(len(accepted_models) / max(1, len(neighbor_models)))

        neighbor_avg = _average_state_dicts(accepted_models)
        aggregated = {}
        for key in own_model.keys():
            aggregated[key] = (
                self.config.balance_alpha * own_model[key]
                + (1 - self.config.balance_alpha) * neighbor_avg[key]
            )
        return aggregated, accepted_indices, rejected_indices
        
class UBARAggregator: 

    """UBAR: Two-stage Byzantine-robust aggregation (distance and performance filtering)."""

    def __init__(self, node_id: int, config: SimulationConfig, train_loader: DataLoader, device: torch.device, model_template:nn.Module):
        self.node_id = node_id
        self.config = config            
        self.train_loader = train_loader
        self.device = device
        self.model_template = model_template
        self.criterion = nn.CrossEntropyLoss()
        
        self.stats = {
            "accepted": 0,
            "total": 0,
            "stage1_rates": [],
            "stage2_rates": []
        }
    
    def aggregate(self, own_model: Dict[str, torch.Tensor],
              neighbor_models: List[Dict[str, torch.Tensor]],
              neighbor_indices: List[int] = None) -> Tuple[Dict[str, torch.Tensor], List[int], List[int]]:
        """Two-stage UBAR aggregation."""
        if not neighbor_models:
            return own_model, [], []
        
        if neighbor_indices is None:
            neighbor_indices = list(range(len(neighbor_models)))

        # Stage 1: Distance-based filtering
        distances = [self._compute_distance(own_model, m) for m in neighbor_models]
        num_select = max(1, int(self.config.ubar_rho * len(neighbor_models)))
        
        sorted_indices = np.argsort(distances)
        stage1_selected = [neighbor_models[i] for i in sorted_indices[:num_select]]
        stage1_nids = [neighbor_indices[i] for i in sorted_indices[:num_select]]
        
        stage1_rate = len(stage1_selected) / max(1, len(neighbor_models))
        self.stats["stage1_rates"].append(stage1_rate)
        
        # Stage 2: Performance-based filtering 
        try:
            sample_batch = next(iter(self.train_loader))
            own_loss = self._compute_loss(own_model, sample_batch)
            
            stage2_selected = []
            stage2_nids = []
            for model, nid in zip(stage1_selected, stage1_nids):
                model_loss = self._compute_loss(model, sample_batch)
                if model_loss <= own_loss:
                    stage2_selected.append(model)
                    stage2_nids.append(nid)
            
            # Fallback
            if not stage2_selected:
                stage2_selected = [stage1_selected[0]]
                stage2_nids = [stage1_nids[0]]
            
            stage2_rate = len(stage2_selected) / max(1, len(stage1_selected))
            self.stats["stage2_rates"].append(stage2_rate)
            
        except StopIteration:
            stage2_selected = stage1_selected
            stage2_nids = stage1_nids
            self.stats["stage2_rates"].append(1.0)
        
        # Statistics
        self.stats["total"] += len(neighbor_models)
        self.stats["accepted"] += len(stage2_selected)
        
        # Compute accepted/rejected indices
        accepted_indices = list(stage2_nids)
        rejected_indices = [nid for nid in neighbor_indices if nid not in accepted_indices]

        # Aggregate
        neighbor_avg = _average_state_dicts(stage2_selected)
        
        aggregated = {}
        for key in own_model.keys():
            aggregated[key] = (self.config.balance_alpha * own_model[key] + 
                            (1 - self.config.balance_alpha) * neighbor_avg[key])
        
        return aggregated, accepted_indices, rejected_indices
    
    def _compute_distance(self, model1: Dict[str, torch.Tensor], model2: Dict[str, torch.Tensor]) -> float:
        total = 0.0
        for key in model1.keys():
            diff = model1[key].float() - model2[key].float()
            total += torch.sum(diff * diff).item()
        return float(np.sqrt(total))
    
    def _compute_loss(self, model: Dict[str, torch.Tensor], batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        self.model_template.load_state_dict(model, strict = False)
        self.model_template.eval()

        x, y = batch 
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            logits = self.model_template(x)
            loss = self.criterion(logits, y)
        
        return loss.item()
        
#Federated Nodes

class FederatedNode: 
    """Represents a single node in the federated network."""

    def __init__(self, node_id: int, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, config: SimulationConfig, device: torch.device, total_rounds:int):
        self.node_id = node_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device 

        # Initialise aggregator 
        if config.aggregation == "fedavg":
            self.aggregator = FedAvgAggregator(node_id)
        elif config.aggregation == "balance": 
            self.aggregator = BALANCEAggregator(node_id, config, total_rounds)
        elif config.aggregation == "ubar":
            self.aggregator = UBARAggregator(node_id, config, train_loader, device, model)
        else:
            raise ValueError(f"Unknown aggregation method: {config.aggregation}")
        
    def train_local(self):
        """Local training: Trains the model on local data and returns the updated state dict"""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate the model on local test data and return accuracy and loss"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0 
        correct = 0 
        total = 0 

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader: 
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

                _, predicted = torch.max(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / max(1, total) 
        avg_loss = total_loss / max(1, total)
        # Guard against NaN / Inf from exploding weights
        if not np.isfinite(avg_loss):
            avg_loss = float("inf")
        return accuracy, avg_loss
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get the model parameters"""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_state(self, state: Dict[str, torch.Tensor]):
        """Set the model parameters"""
        self.model.load_state_dict(state, strict = False)

# Main Simulator 

class DecentralisedSimulator:
    """Main simulator for decentralised federated learning"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._set_seed()
        self.device = self._get_device()
        
        # Load dataset
        self._load_data()
        
        # Create network graph
        self.graph = NetworkGraph(config.num_nodes, config.topology, config.k_neighbors)
        
        # Initialize nodes
        self._initialize_nodes()
        
        # Initialize attacker
        self.attacker = None
        if config.attack_ratio > 0:
            self.attacker = ByzantineAttacker(config)
        
        # Results tracking
        self.results = {
            "accuracies": [],  # Per round, per node
            "losses": [],
            "acceptance_rates": [],
            "honest_accuracies": [],
            "compromised_accuracies": [],
            #- Advanced metrics-
            "drift_per_round": [],               # {mean, std, per_node}
            "peer_deviation_per_round": [],       # {mean, per_node}
            "consensus_score_per_round": [],      # float
            "regression_slope_per_round": [],     # {slope, r_squared}
            "detection_flags_per_round": [],      # [{node_id, drift, z_score, anomaly_score, flagged}]
            "overhead_time": {"with_detection": [], "without_detection": []},
        }
        # Confusion-matrix accumulators
        self._detection_tp = 0
        self._detection_fp = 0
        self._detection_tn = 0
        self._detection_fn = 0
        self._detection_time = None   # first round a compromised node is correctly flagged
        self._previous_states = None  # previous-round model states for drift

        # Verification layer persistent state
        self._trust_scores = [self.config.trust_initial] * self.config.num_nodes
        self._drift_history = [[] for _ in range(self.config.num_nodes)]
        self._peer_dev_history = [[] for _ in range(self.config.num_nodes)]
        self._verification_flags = []  # per-round verification flags
        self._verification_flags_per_round = []  # list of per-round flag lists
        self._verification_time_per_round = []  # per-round verification time
        self._round_accepted = []  # per-node accepted sets (set each round)
        self._round_rejected = []  # per-node rejected sets (set each round)
        self._round_neighbor_models = {}  # (receiver_idx, sender_idx) -> raw state_dict sent this round
        self._round_message_stats = {}  # (receiver_idx, sender_idx) -> message-level verification features
        self._verification_suspicion_counts = {}  # (receiver_idx, sender_idx) -> consecutive suspicious rounds
        self._verification_warmup_rounds = max(
            3, min(self.config.num_rounds, self.config.verification_history_window)
        )
    
    def _set_seed(self):
        """Set random seeds."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _get_device(self) -> torch.device:
        """Get compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _load_data(self):
        """Load and partition dataset."""
        data_path = f"./leaf/data/{self.config.dataset}/data"
        
        train_ds, test_ds, model_template, num_classes, input_size = \
            load_leaf_dataset(self.config.dataset, data_path)
        
        self.train_dataset = train_ds
        self.test_dataset = test_ds
        self.model_template = model_template
        self.num_classes = num_classes
        
        # Partition data
        train_partitions, test_partitions = create_leaf_client_partitions(
            train_ds, test_ds, self.config.num_nodes, self.config.seed,
            alpha=self.config.partition_alpha
        )
        
        self.train_partitions = train_partitions
        self.test_partitions = test_partitions
        
        print(f"Loaded {self.config.dataset} dataset")
        print(f"   Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    def _initialize_nodes(self):
        """Initialize federated nodes."""
        self.nodes = []
        
        for i in range(self.config.num_nodes):
            # Create model
            torch.manual_seed(self.config.seed + i)
            if self.config.dataset == "femnist":
                from leaf_datasets import LEAFFEMNISTModel
                model = LEAFFEMNISTModel(self.num_classes).to(self.device)
            elif self.config.dataset == "celeba":
                # Future: CelebA is supported programmatically but not yet exposed via CLI
                from leaf_datasets import LEAFCelebAModel
                model = LEAFCelebAModel(self.num_classes).to(self.device)
            elif self.config.dataset == "shakespeare":
                from leaf_datasets import LEAFShakespeareModel
                model = LEAFShakespeareModel(self.num_classes).to(self.device)
            else:
                raise ValueError(f"Unsupported dataset: {self.config.dataset}")
            
            # Create data loaders
            train_subset = Subset(self.train_dataset, self.train_partitions[i])
            test_subset = Subset(self.test_dataset, self.test_partitions[i])
            
            train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, 
                                     shuffle=True, num_workers=0)
            test_loader = DataLoader(test_subset, batch_size=512, 
                                    shuffle=False, num_workers=0)
            
            # Create node
            node = FederatedNode(i, model, train_loader, test_loader, 
                               self.config, self.device, self.config.num_rounds)
            self.nodes.append(node)
        
        print(f"Initialized {self.config.num_nodes} nodes")
        print(f"   Topology: {self.config.topology}")
        print(f"   Aggregation: {self.config.aggregation}")

   
    #  Advanced per-round metrics
    

    def _compute_drift(self, current_states, previous_states):
        """L2 norm of parameter change per node between consecutive rounds."""
        drifts = []
        for curr, prev in zip(current_states, previous_states):
            diff_norm_sq = 0.0
            for key in curr.keys():
                diff = curr[key].float() - prev[key].float()
                diff_norm_sq += torch.sum(diff * diff).item()
            val = float(np.sqrt(diff_norm_sq))
            drifts.append(val if np.isfinite(val) else 0.0)
        return drifts

    def _compute_peer_deviation(self, all_states):
        """Mean L2 distance of each node to its neighbours; global consensus."""
        deviations = []
        for node_idx in range(self.config.num_nodes):
            neighbors = self.graph.get_neighbors(node_idx)
            own = all_states[node_idx]
            dists = []
            for n_idx in neighbors:
                d = 0.0
                for key in own.keys():
                    diff = own[key].float() - all_states[n_idx][key].float()
                    d += torch.sum(diff * diff).item()
                val = float(np.sqrt(d))
                dists.append(val if np.isfinite(val) else 0.0)
            deviations.append(float(np.mean(dists)) if dists else 0.0)
        mean_dev = float(np.mean(deviations))
        consensus = 1.0 / (1.0 + mean_dev)
        return deviations, consensus

    def _compute_regression_slope(self, values, window=10):
        """OLS slope + R² over the last *window* entries."""
        if len(values) < 2:
            return 0.0, 0.0
        w = values[-window:]
        n = len(w)
        x = np.arange(n, dtype=float)
        y = np.array(w, dtype=float)
        x_m, y_m = np.mean(x), np.mean(y)
        ss_xy = float(np.sum((x - x_m) * (y - y_m)))
        ss_xx = float(np.sum((x - x_m) ** 2))
        ss_yy = float(np.sum((y - y_m) ** 2))
        if ss_xx == 0:
            return 0.0, 0.0
        slope = ss_xy / ss_xx
        r_sq = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0
        return float(slope), float(r_sq)

    def _detect_anomalies(self, drifts, round_num):
        """Z-score based anomaly detection; returns per-node flag dicts."""
        if not drifts or len(drifts) < 2:
            return []
        mean_d = float(np.mean(drifts))
        std_d = float(np.std(drifts))
        flags = []
        for node_id, drift in enumerate(drifts):
            z = (drift - mean_d) / std_d if std_d > 1e-12 else 0.0
            anomaly_score = float(abs(z))
            flagged = anomaly_score > 2.0  # 2-sigma threshold
            flags.append({
                "node_id": node_id,
                "drift": float(drift),
                "z_score": float(z),
                "anomaly_score": anomaly_score,
                "flagged": flagged,
            })
        return flags

    def _update_confusion_matrix(self, flags, round_num):
        """Compare flagged nodes against ground-truth compromised set."""
        if not self.attacker or not flags:
            return
        for f in flags:
            nid = f["node_id"]
            is_compromised = self.attacker.is_compromised(nid)
            is_flagged = f["flagged"]
            if is_compromised and is_flagged:
                self._detection_tp += 1
                if self._detection_time is None:
                    self._detection_time = round_num
            elif not is_compromised and is_flagged:
                self._detection_fp += 1
            elif is_compromised and not is_flagged:
                self._detection_fn += 1
            else:
                self._detection_tn += 1

    def _print_results_table_header(self):
        """Print round metrics table header."""
        if self.attacker:
            header = (
                f"{'Round':>5} | {'Avg Acc':>8} | {'Std Acc':>8} | {'Avg Loss':>8} | "
                f"{'Honest':>8} | {'Compromised':>11}"
            )
        else:
            header = f"{'Round':>5} | {'Avg Acc':>8} | {'Std Acc':>8} | {'Avg Loss':>8}"
        print("\nRound Metrics")
        print(header)
        print("-" * len(header))
    
    def run(self):
        """Run simulation."""
        print("\nStarting simulation")
        self._print_results_table_header()
        
        # Initial evaluation
        self._evaluate_round(0)

        # Store initial states for drift computation
        self._previous_states = [node.get_model_state() for node in self.nodes]

        #Round-0 baseline advanced metrics
        peer_devs_0, consensus_0 = self._compute_peer_deviation(self._previous_states)
        self.results["drift_per_round"].append(
            {"mean": 0.0, "std": 0.0, "per_node": [0.0] * self.config.num_nodes})
        self.results["peer_deviation_per_round"].append(
            {"mean": float(np.mean(peer_devs_0)), "per_node": peer_devs_0})
        self.results["consensus_score_per_round"].append(consensus_0)
        avg_accs_0 = [float(np.mean(a)) for a in self.results["accuracies"]]
        slope_0, r_sq_0 = self._compute_regression_slope(avg_accs_0, window=10)
        self.results["regression_slope_per_round"].append(
            {"slope": slope_0, "r_squared": r_sq_0})
        self.results["detection_flags_per_round"].append([])
        self.results["overhead_time"]["without_detection"].append(0.0)
        self.results["overhead_time"]["with_detection"].append(0.0)
        self._verification_flags_per_round.append([])
        self._verification_time_per_round.append(0.0)

        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            #- Local training
            for node in self.nodes:
                node.train_local()
            
            # Aggregation (timed without detection) 
            t_agg_start = time.time()
            
            pre_agg_states = [node.get_model_state() for node in self.nodes]
            self._aggregation_round(round_num)
            t_agg_end = time.time()
            time_without_detection = t_agg_end - t_agg_start

            # Collect current model states 
            current_states = [node.get_model_state() for node in self.nodes]

            # Drift computation 
            drifts = self._compute_drift(current_states, self._previous_states)
            mean_drift = float(np.mean(drifts))
            std_drift = float(np.std(drifts))
            self.results["drift_per_round"].append({
                "mean": mean_drift, "std": std_drift, "per_node": drifts
            })

            #- Peer deviation + consensus
            peer_devs, consensus = self._compute_peer_deviation(current_states)
            self.results["peer_deviation_per_round"].append({
                "mean": float(np.mean(peer_devs)), "per_node": peer_devs
            })
            self.results["consensus_score_per_round"].append(consensus)

            #- Anomaly detection (timed) — BEFORE verification
            t_det_start = time.time()
            flags = self._detect_anomalies(drifts, round_num)
            self._update_confusion_matrix(flags, round_num)
            t_det_end = time.time()
            detection_overhead = t_det_end - t_det_start
            self.results["detection_flags_per_round"].append(flags)

            z_scores = [f["z_score"] for f in flags] if flags else [0.0] * self.config.num_nodes

            
            # VERIFICATION LAYER (post-acceptance)     
            # Uses drifts, peer deviations, z-scores, and pre-aggregation states
            # Can flag nodes for isolation or rescue based on configurable logic
            self._verification_flags = []  # reset per-round
            t_ver_start = time.time()
            verification_changed = self._run_verification(
                round_num, current_states, drifts, peer_devs, z_scores, pre_agg_states
            )
            t_ver_end = time.time()
            verification_time = t_ver_end - t_ver_start

            # Store verification results for this round
            self._verification_flags_per_round.append(list(self._verification_flags))
            self._verification_time_per_round.append(verification_time)

            # If verification changed models, re-collect states
            if verification_changed:
                current_states = [node.get_model_state() for node in self.nodes]

            # Evaluation (AFTER verification) 
            self._evaluate_round(round_num)

            # Linear regression slope (plotted over accumulated avg accuracies)
            avg_accs = [float(np.mean(a)) for a in self.results["accuracies"]]
            slope, r_sq = self._compute_regression_slope(avg_accs, window=10)
            self.results["regression_slope_per_round"].append({
                "slope": slope, "r_squared": r_sq
            })

            # Overhead time 
            time_with_all = time_without_detection + detection_overhead + verification_time
            self.results["overhead_time"]["without_detection"].append(
                time_without_detection)
            self.results["overhead_time"]["with_detection"].append(
                time_with_all)

            # Structured metrics line 
            n_flagged = sum(1 for f in flags if f["flagged"]) if flags else 0
            n_ver_flagged = sum(1 for vf in self._verification_flags if vf["action"] == "flagged")
            n_ver_rescued = sum(1 for vf in self._verification_flags if vf["action"] == "rescued")
            print(
                f"METRICS|{round_num}|{mean_drift:.6f}|{std_drift:.6f}"
                f"|{float(np.mean(peer_devs)):.6f}|{consensus:.6f}"
                f"|{slope:.6f}|{r_sq:.6f}"
                f"|{n_flagged}|{self._detection_tp}|{self._detection_fp}"
                f"|{self._detection_tn}|{self._detection_fn}"
                f"|{time_without_detection:.6f}|{time_with_all:.6f}"
                f"|{n_ver_flagged}|{n_ver_rescued}|{verification_time:.6f}"
            )

            # Store current states for next round's drift
            self._previous_states = current_states
        
        # Final summary
        self._print_summary()
        
        return self.results
    
    def _aggregation_round(self, round_num: int):
        """Perform one aggregation round."""
        # Collect all model states
        all_states = [node.get_model_state() for node in self.nodes]
        
        # Each node aggregates with neighbors
        new_states = []
        self._round_accepted = []
        self._round_rejected = []
        self._round_neighbor_models = {}  # reset each round
        
        for node_idx, node in enumerate(self.nodes):
            neighbors_idx = self.graph.get_neighbors(node_idx)
            own_state = all_states[node_idx]
            
            # Get neighbor states (potentially malicious)
            neighbor_states = []
            for neighbor_idx in neighbors_idx:
                if self.attacker and self.attacker.is_compromised(neighbor_idx):
                    # Craft malicious update
                    honest_neighbors = [i for i in neighbors_idx 
                                      if not self.attacker.is_compromised(i)]
                    honest_states = [all_states[i] for i in honest_neighbors]
                    malicious_state = self.attacker.craft_malicious_update(
                        honest_states, neighbor_idx
                    )
                    raw_state = malicious_state if malicious_state else all_states[neighbor_idx]
                else:
                    raw_state = all_states[neighbor_idx]
               
                self._round_neighbor_models[(node_idx, neighbor_idx)] = raw_state
                neighbor_states.append(raw_state)
            
            # Aggregate — now returns accepted/rejected indices
            new_state, accepted, rejected = node.aggregator.aggregate(
                own_state, neighbor_states, neighbors_idx
            )
            new_states.append(new_state)
            self._round_accepted.append(accepted)
            self._round_rejected.append(rejected)
        
        # Update all nodes
        for node, new_state in zip(self.nodes, new_states):
            node.set_model_state(new_state)
  
    # Verification Layer
    _PHASE2_MIN_ACCURACY = 0.05
    _FLAG_CONSECUTIVE_REQUIRED = 2

    

    def _evaluate_model(self, node_idx, model_state):
        """
        Evaluate a model state dict on node's LOCAL TEST SET.
        Returns: float — accuracy in range [0.0, 1.0].
        IMPORTANT: This must NOT modify the node's actual model.
        """
        node = self.nodes[node_idx]
        original_state = node.get_model_state()  # save
        node.set_model_state(model_state)
        acc, _ = node.evaluate()
        node.set_model_state(original_state)     # restore
        return acc

    def _re_aggregate(self, node_idx, accepted_models, pre_agg_state=None):
        """
        Re-aggregate using the same blending formula as the aggregation algorithm.
        θ_new = α · θ_own + (1 - α) · mean(accepted_models)
        pre_agg_state: the node's locally-trained state before this round's aggregation.
        When provided, it is used as θ_own to avoid double-counting aggregation.
        """
        if not accepted_models:
            return self.nodes[node_idx].get_model_state()

        alpha = self.config.balance_alpha
        #use the pre-aggregation state as own_model when available
        own_model = pre_agg_state if pre_agg_state is not None else self.nodes[node_idx].get_model_state()

        # Average accepted models
        avg = {}
        for key in accepted_models[0].keys():
            stacked = torch.stack([m[key].float() for m in accepted_models])
            avg[key] = stacked.mean(dim=0).to(accepted_models[0][key].dtype)

        # Convex blend
        result = {}
        for key in own_model.keys():
            result[key] = alpha * own_model[key] + (1 - alpha) * avg[key]

        return result

    def _get_neighbor_model(self, node_idx, neighbor_idx, fallback_states):
        """Return the raw model that ``neighbor_idx`` sent to ``node_idx`` this round."""
        key = (node_idx, neighbor_idx)
        if key in self._round_neighbor_models:
            return self._round_neighbor_models[key]
        return fallback_states[neighbor_idx]

    def _state_distance(self, model1: Dict[str, torch.Tensor], model2: Dict[str, torch.Tensor]) -> float:
        """Compute the L2 distance between two model state dicts."""
        total = 0.0
        for key in model1.keys():
            diff = model1[key].float() - model2[key].float()
            total += torch.sum(diff * diff).item()
        return float(np.sqrt(total))

    def _message_z_scores(self, values: List[float]) -> List[float]:
        """Return z-scores for a list of scalar message features."""
        if not values:
            return []
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        if std_v <= 1e-12:
            return [0.0 for _ in values]
        return [float((v - mean_v) / std_v) for v in values]

    def _build_message_stats(self, node_idx: int, accepted_ids: List[int], rejected_ids: List[int],
                             pre_agg_state: Dict[str, torch.Tensor], fallback_states: List[Dict[str, torch.Tensor]]) -> Dict[int, Dict[str, float]]:
        """
        Build message-level verification features for every neighbour of ``node_idx``.

        Features are derived from the actual raw message sent this round, not from the
        sender's final post-aggregation node state.
        """
        neighbor_ids = list(accepted_ids) + list(rejected_ids)
        if not neighbor_ids:
            return {}

        messages = {
            nid: self._get_neighbor_model(node_idx, nid, fallback_states)
            for nid in neighbor_ids
        }
        distances_to_own = {
            nid: self._state_distance(pre_agg_state, msg_state)
            for nid, msg_state in messages.items()
        }
        values = list(distances_to_own.values())
        z_scores = self._message_z_scores(values)
        z_by_neighbor = {
            nid: z_scores[idx]
            for idx, nid in enumerate(distances_to_own.keys())
        }

        stats = {}
        for nid in neighbor_ids:
            other_models = [messages[oid] for oid in neighbor_ids if oid != nid]
            if other_models:
                median_like = _average_state_dicts(other_models)
                peer_distance = self._state_distance(messages[nid], median_like)
            else:
                peer_distance = 0.0

            stats[nid] = {
                "distance_to_own": float(distances_to_own[nid]),
                "peer_distance": float(peer_distance),
                "z_score": float(z_by_neighbor.get(nid, 0.0)),
            }
        return stats

    def _verification_phase1(self, node_idx, S, R, theta_agg, acc_agg,
                              z_scores, original_states, round_num, pre_agg_state=None):
        """
        Phase 1: conservatively remove already-accepted neighbours whose *messages*
        look persistently suspicious and whose removal measurably improves validation.
        """
        if round_num < self._verification_warmup_rounds:
            return False

        eps = self.config.verification_epsilon
        msg_stats = self._build_message_stats(node_idx, S, R, pre_agg_state, original_states)
        if not msg_stats:
            return False

        original_S = list(S)
        nodes_to_flag = []

        hist_ready = len([h for h in self._drift_history if h]) >= 1
        drift_means = [
            sum(hist) / len(hist) for hist in self._drift_history if hist
        ]
        peer_means = [
            sum(hist) / len(hist) for hist in self._peer_dev_history if hist
        ]
        pop_mean_drift = float(np.mean(drift_means)) if drift_means else 0.0
        pop_mean_peer = float(np.mean(peer_means)) if peer_means else 0.0

        for j in list(S):
            message_info = msg_stats.get(j)
            if not message_info:
                continue

            msg_z = abs(message_info["z_score"])
            if not (self.config.z_low <= msg_z <= self.config.z_high):
                self._verification_suspicion_counts[(node_idx, j)] = 0
                continue

            s_without_j = [k for k in original_S if k != j]
            if not s_without_j:
                self._verification_suspicion_counts[(node_idx, j)] = 0
                continue

            models_without_j = [
                self._get_neighbor_model(node_idx, k, original_states)
                for k in s_without_j
            ]
            theta_without_j = self._re_aggregate(node_idx, models_without_j, pre_agg_state)
            acc_without_j = self._evaluate_model(node_idx, theta_without_j)
            delta_perf = acc_without_j - acc_agg

            sig_low_trust = self._trust_scores[j] < 0.35
            sig_high_drift = bool(
                self._drift_history[j] and
                (sum(self._drift_history[j]) / len(self._drift_history[j])) > pop_mean_drift
            )
            sig_high_pdev = bool(
                self._peer_dev_history[j] and
                (sum(self._peer_dev_history[j]) / len(self._peer_dev_history[j])) > pop_mean_peer
            )
            sig_message_far = message_info["peer_distance"] >= message_info["distance_to_own"]

            signals = sum([sig_low_trust, sig_high_drift, sig_high_pdev, sig_message_far])

            suspicious = delta_perf > eps and signals >= 3 and hist_ready
            suspicion_key = (node_idx, j)
            if suspicious:
                self._verification_suspicion_counts[suspicion_key] = (
                    self._verification_suspicion_counts.get(suspicion_key, 0) + 1
                )
            else:
                self._verification_suspicion_counts[suspicion_key] = 0

            if self._verification_suspicion_counts[suspicion_key] < self._FLAG_CONSECUTIVE_REQUIRED:
                continue

            nodes_to_flag.append((j, delta_perf, message_info, sig_low_trust, sig_high_drift, sig_high_pdev, sig_message_far))

        changed = False
        for (j, delta_perf, message_info, sig_low_trust, sig_high_drift, sig_high_pdev, sig_message_far) in nodes_to_flag:
            if j not in S:
                continue
            trust_before = self._trust_scores[j]
            S.remove(j)
            if j not in R:
                R.append(j)
            self._trust_scores[j] = max(0.0, self._trust_scores[j] - self.config.trust_penalty)
            self._verification_suspicion_counts[(node_idx, j)] = 0
            changed = True

            self._verification_flags.append({
                "node_id": j,
                "target_node": node_idx,
                "phase": 1,
                "action": "flagged",
                "delta_perf": float(delta_perf),
                "trust_before": float(trust_before),
                "trust_after": float(self._trust_scores[j]),
                "message_stats": message_info,
                "signals": {
                    "low_trust": sig_low_trust,
                    "high_drift": sig_high_drift,
                    "high_peer_dev": sig_high_pdev,
                    "message_far_from_peers": sig_message_far,
                },
            })

        return changed

    def _verification_phase2(self, node_idx, S, R, theta_agg, acc_agg,
                              z_scores, drifts, peer_devs, original_states, round_num, pre_agg_state=None):
        """
        Phase 2: rescue previously rejected neighbours when their message looks benign
        and adding them back does not materially hurt validation accuracy.
        """
        if acc_agg < self._PHASE2_MIN_ACCURACY:
            return False

        eps = self.config.verification_epsilon
        msg_stats = self._build_message_stats(node_idx, S, R, pre_agg_state, original_states)
        if not msg_stats:
            return False

        sorted_pdevs = sorted(peer_devs)
        median_peer_dev = float(sorted_pdevs[len(sorted_pdevs) // 2]) if sorted_pdevs else 0.0
        mu_drift = float(np.mean(drifts)) if drifts else 0.0
        sigma_drift = float(np.std(drifts)) if drifts else 0.0

        changed = False
        for k in list(R):
            message_info = msg_stats.get(k)
            if not message_info:
                continue

            sig_trust_ok = self._trust_scores[k] >= 0.4
            sig_pdev_ok = peer_devs[k] <= median_peer_dev if peer_devs else True
            sig_drift_ok = drifts[k] <= mu_drift + 1.0 * sigma_drift if drifts else True
            sig_message_ok = abs(message_info["z_score"]) < self.config.z_high

            s_with_k = S + [k]
            models_with_k = [self._get_neighbor_model(node_idx, j, original_states) for j in s_with_k]
            theta_with_k = self._re_aggregate(node_idx, models_with_k, pre_agg_state)
            acc_with_k = self._evaluate_model(node_idx, theta_with_k)
            sig_perf_ok = acc_with_k >= acc_agg - eps

            signals = sum([sig_trust_ok, sig_pdev_ok, sig_drift_ok, sig_message_ok, sig_perf_ok])
            if signals >= 4 and sig_perf_ok:
                R.remove(k)
                if k not in S:
                    S.append(k)
                old_trust = self._trust_scores[k]
                self._trust_scores[k] = min(1.0, self._trust_scores[k] + self.config.trust_boost)
                self._verification_suspicion_counts[(node_idx, k)] = 0
                changed = True
                self._verification_flags.append({
                    "node_id": k,
                    "target_node": node_idx,
                    "phase": 2,
                    "action": "rescued",
                    "delta_perf": float(acc_with_k - acc_agg),
                    "trust_before": float(old_trust),
                    "trust_after": float(self._trust_scores[k]),
                    "message_stats": message_info,
                    "signals": {
                        "trust_ok": sig_trust_ok,
                        "peer_dev_ok": sig_pdev_ok,
                        "drift_ok": sig_drift_ok,
                        "message_ok": sig_message_ok,
                        "performance_ok": sig_perf_ok,
                    },
                })
        return changed

    def _verification_phase4(self, round_num, z_scores):
        """Update node trust using conservative round-level anomaly evidence."""
        beta = self.config.trust_decay
        modified_nodes = {
            flag["node_id"]
            for flag in self._verification_flags
            if flag.get("action") in ("flagged", "rescued")
        }

        for i in range(self.config.num_nodes):
            if i in modified_nodes:
                continue
            was_flagged = abs(z_scores[i]) > self.config.z_high
            target = 0.0 if was_flagged else 1.0
            self._trust_scores[i] = beta * self._trust_scores[i] + (1 - beta) * target

    def _run_verification(self, round_num, all_states, drifts, peer_devs, z_scores, pre_agg_states):
        """
        Called once per round, after aggregation and first-pass detection.
        pre_agg_states: per-node model states captured before _aggregation_round().
        Returns: corrected: bool — whether any changes were made.
        """
        if not self.config.verification_enabled:
            return False
        if self.config.aggregation == "fedavg":
            return False

        # Update history buffers (BEFORE running phases)
        w = self.config.verification_history_window
        for i in range(self.config.num_nodes):
            self._drift_history[i].append(drifts[i])
            if len(self._drift_history[i]) > w:
                self._drift_history[i].pop(0)
            self._peer_dev_history[i].append(peer_devs[i])
            if len(self._peer_dev_history[i]) > w:
                self._peer_dev_history[i].pop(0)

        any_changed = False
        original_states = list(all_states)
        # Collect all model updates; apply them after the full loop.
        pending_updates = {}

        for node_idx in range(self.config.num_nodes):
            S = list(self._round_accepted[node_idx])  # copy — will be modified
            R = list(self._round_rejected[node_idx])  # copy

            if not R and not S:
                continue  # nothing to verify

            theta_agg = original_states[node_idx]  # initial aggregated model for this node
            acc_agg = self._evaluate_model(node_idx, theta_agg)

            pre_agg_state = pre_agg_states[node_idx] if pre_agg_states else None
            changed = False

            #PHASE 1
            changed_p1 = self._verification_phase1(
                node_idx, S, R, theta_agg, acc_agg,
                z_scores, original_states, round_num, pre_agg_state
            )
            if changed_p1:
                changed = True

            # PHASE 2 
            changed_p2 = self._verification_phase2(
                node_idx, S, R, theta_agg, acc_agg,
                z_scores, drifts, peer_devs, original_states, round_num, pre_agg_state
            )
            if changed_p2:
                changed = True

            # PHASE 3: Re-aggregate if anything changed
            if changed:
                
                accepted_models = [self._get_neighbor_model(node_idx, j, original_states) for j in S]
                if accepted_models:
                    theta_final = self._re_aggregate(node_idx, accepted_models, pre_agg_state)
                    
                    pending_updates[node_idx] = theta_final
                any_changed = True

            # Update accepted/rejected records
            self._round_accepted[node_idx] = S
            self._round_rejected[node_idx] = R

        for idx, state in pending_updates.items():
            self.nodes[idx].set_model_state(state)
            all_states[idx] = state

        # Phase 4: trust updates
        self._verification_phase4(round_num, z_scores)

        return any_changed

    def _evaluate_round(self, round_num: int):
        """Evaluate all nodes."""
        accuracies = []
        losses = []
        
        for node in self.nodes:
            acc, loss = node.evaluate()
            accuracies.append(acc)
            losses.append(loss)
        
        self.results["accuracies"].append(accuracies)
        self.results["losses"].append(losses)

        avg_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        avg_loss = float(np.mean(losses))
        
        # Split by honest/compromised
        if self.attacker:
            honest_accs = [accuracies[i] for i in range(self.config.num_nodes) 
                          if not self.attacker.is_compromised(i)]
            compromised_accs = [accuracies[i] for i in range(self.config.num_nodes) 
                               if self.attacker.is_compromised(i)]
            
            honest_mean = float(np.mean(honest_accs)) if honest_accs else 0.0
            compromised_mean = float(np.mean(compromised_accs)) if compromised_accs else 0.0

            self.results["honest_accuracies"].append(honest_mean)
            self.results["compromised_accuracies"].append(compromised_mean)

            print(
                f"{round_num:>5} | {avg_acc:>8.4f} | {std_acc:>8.4f} | {avg_loss:>8.4f} | "
                f"{honest_mean:>8.4f} | {compromised_mean:>11.4f}"
            )
        else:
            print(f"{round_num:>5} | {avg_acc:>8.4f} | {std_acc:>8.4f} | {avg_loss:>8.4f}")
    
    def _print_summary(self):
        """Print final summary."""
        print("\nFinal Results")
        
        final_accs = self.results["accuracies"][-1]
        print(f"Final Accuracy: {np.mean(final_accs):.4f} +/- {np.std(final_accs):.4f}")
        
        if self.attacker:
            honest_acc = self.results["honest_accuracies"][-1]
            comp_acc = self.results["compromised_accuracies"][-1]
            print(f"Honest Nodes: {honest_acc:.4f}")
            print(f"Compromised Nodes: {comp_acc:.4f}")
            print(f"Attack Impact: {honest_acc - comp_acc:.4f}")

        #Advanced metrics summary
        if self.results["drift_per_round"]:
            last_drift = self.results["drift_per_round"][-1]
            print(f"\nDrift (last round):  mean={last_drift['mean']:.6f}  std={last_drift['std']:.6f}")

        if self.results["consensus_score_per_round"]:
            print(f"Consensus Score:     {self.results['consensus_score_per_round'][-1]:.6f}")

        if self.results["regression_slope_per_round"]:
            last_reg = self.results["regression_slope_per_round"][-1]
            print(f"Regression Slope:    slope={last_reg['slope']:.6f}  R²={last_reg['r_squared']:.6f}")

        if self.attacker:
            print(f"\nDetection Metrics:")
            print(f"  True Positives:   {self._detection_tp}")
            print(f"  False Positives:  {self._detection_fp}")
            print(f"  True Negatives:   {self._detection_tn}")
            print(f"  False Negatives:  {self._detection_fn}")
            if self._detection_time is not None:
                print(f"  Detection Time:   T_detect = round {self._detection_time}")
            else:
                print(f"  Detection Time:   not detected")

        if self.results["overhead_time"]["with_detection"]:
            avg_wo = float(np.mean(self.results["overhead_time"]["without_detection"]))
            avg_w  = float(np.mean(self.results["overhead_time"]["with_detection"]))
            print(f"\nOverhead (avg per round):")
            print(f"  Without detection: {avg_wo:.4f}s")
            print(f"  With detection:    {avg_w:.4f}s")

        #Verification layer summary
        if self.config.verification_enabled and self.config.aggregation != "fedavg":
            total_p1 = sum(
                sum(1 for vf in rflags if vf["action"] == "flagged")
                for rflags in self._verification_flags_per_round
            )
            total_p2 = sum(
                sum(1 for vf in rflags if vf["action"] == "rescued")
                for rflags in self._verification_flags_per_round
            )
            print(f"\nVerification Layer:")
            print(f"  Phase 1 flags (total): {total_p1}")
            print(f"  Phase 2 rescues (total): {total_p2}")
            print(f"  Final trust scores: {[round(t, 3) for t in self._trust_scores]}")
    
    def save_results(self, filepath: str):
        """Save results to JSON with run timestamp and per-round table."""
        from datetime import datetime as _dt
        now = _dt.now()

        # Build tabular round-by-round rows
        round_results = []
        for r in range(len(self.results["accuracies"])):
            accs   = self.results["accuracies"][r]
            losses = self.results["losses"][r]
            row = {
                "round":                r,
                "avg_accuracy":        float(np.mean(accs)),
                "std_accuracy":        float(np.std(accs)),
                "avg_loss":            float(np.mean(losses)),
                "honest_accuracy":     (
                    float(self.results["honest_accuracies"][r])
                    if self.attacker and r < len(self.results["honest_accuracies"])
                    else None
                ),
                "compromised_accuracy": (
                    float(self.results["compromised_accuracies"][r])
                    if self.attacker and r < len(self.results["compromised_accuracies"])
                    else None
                ),
            }
            # Advanced metrics (now exist from r == 0)
            adv_idx = r
            if adv_idx < len(self.results["drift_per_round"]):
                dr = self.results["drift_per_round"][adv_idx]
                row["drift_mean"] = dr["mean"]
                row["drift_std"]  = dr["std"]
            if adv_idx < len(self.results["peer_deviation_per_round"]):
                row["peer_deviation_mean"] = self.results["peer_deviation_per_round"][adv_idx]["mean"]
            if adv_idx < len(self.results["consensus_score_per_round"]):
                row["consensus_score"] = self.results["consensus_score_per_round"][adv_idx]
            if adv_idx < len(self.results["regression_slope_per_round"]):
                reg = self.results["regression_slope_per_round"][adv_idx]
                row["regression_slope"] = reg["slope"]
                row["regression_r_squared"] = reg["r_squared"]
            if adv_idx < len(self.results["detection_flags_per_round"]):
                row["detection_flags"] = self.results["detection_flags_per_round"][adv_idx]
            if adv_idx < len(self.results["overhead_time"]["without_detection"]):
                row["time_without_detection"] = self.results["overhead_time"]["without_detection"][adv_idx]
                row["time_with_detection"]    = self.results["overhead_time"]["with_detection"][adv_idx]
            # Verification layer per-round data
            if adv_idx < len(self._verification_flags_per_round):
                vflags = self._verification_flags_per_round[adv_idx]
                row["verification_flags"] = vflags
                row["verification_time"] = self._verification_time_per_round[adv_idx] if adv_idx < len(self._verification_time_per_round) else 0.0
                row["nodes_flagged_phase1"] = sum(1 for vf in vflags if vf.get("action") == "flagged")
                row["nodes_rescued_phase2"] = sum(1 for vf in vflags if vf.get("action") == "rescued")
            round_results.append(row)

        final_accs = self.results["accuracies"][-1]
        summary = {
            "final_accuracy":     float(np.mean(final_accs)),
            "final_accuracy_std": float(np.std(final_accs)),
            "honest_accuracy": (
                float(self.results["honest_accuracies"][-1])
                if self.attacker else None
            ),
            "compromised_accuracy": (
                float(self.results["compromised_accuracies"][-1])
                if self.attacker else None
            ),
            "attack_impact": (
                float(self.results["honest_accuracies"][-1]
                      - self.results["compromised_accuracies"][-1])
                if self.attacker else None
            ),
            # Detection confusion matrix
            "detection": {
                "true_positives":  self._detection_tp,
                "false_positives": self._detection_fp,
                "true_negatives":  self._detection_tn,
                "false_negatives": self._detection_fn,
                "detection_time":  self._detection_time,
                "precision": self._detection_tp / max(1, self._detection_tp + self._detection_fp),
                "recall": self._detection_tp / max(1, self._detection_tp + self._detection_fn),
                "f1_score": (
                    2 * (self._detection_tp / max(1, self._detection_tp + self._detection_fp))
                      * (self._detection_tp / max(1, self._detection_tp + self._detection_fn))
                    / max(1e-12,
                          (self._detection_tp / max(1, self._detection_tp + self._detection_fp))
                        + (self._detection_tp / max(1, self._detection_tp + self._detection_fn)))
                ),
            } if self.attacker else None,
            # Overhead timing
            "overhead_avg": {
                "without_detection": (
                    float(np.mean(self.results["overhead_time"]["without_detection"]))
                    if self.results["overhead_time"]["without_detection"] else None
                ),
                "with_detection": (
                    float(np.mean(self.results["overhead_time"]["with_detection"]))
                    if self.results["overhead_time"]["with_detection"] else None
                ),
            },
            # Verification layer summary
            "verification": {
                "total_phase1_flags": sum(
                    sum(1 for vf in rflags if vf.get("action") == "flagged")
                    for rflags in self._verification_flags_per_round
                ),
                "total_phase2_rescues": sum(
                    sum(1 for vf in rflags if vf.get("action") == "rescued")
                    for rflags in self._verification_flags_per_round
                ),
                "final_trust_scores": [round(t, 6) for t in self._trust_scores],
            } if self.config.verification_enabled and self.config.aggregation != "fedavg" else None,
        }

        output = {
            "run_timestamp": now.isoformat(timespec='seconds'),
            "run_date":      now.strftime('%Y-%m-%d'),
            "config": {
                "dataset":      self.config.dataset,
                "num_nodes":    self.config.num_nodes,
                "num_rounds":   self.config.num_rounds,
                "topology":     self.config.topology,
                "aggregation":  self.config.aggregation,
                "attack_ratio": self.config.attack_ratio,
                "attack_type":  self.config.attack_type,
            },
            # Tabular round results displayed in the terminal (human-readable)
            "round_results": round_results,
            # High-level summary
            "summary": summary,
            # Raw arrays retained for visualisations.py backward compatibility
            "results": self.results,
            "compromised_nodes": (
                list(self.attacker.compromised_nodes) if self.attacker else []
            ),
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filepath}")

        # Export Google Sheets if sheets_config.json is present
        try:
            from sheets_exporter import export_results as _sheets_export
            _sheets_export(filepath)
        except Exception:
            pass  # Sheets export is always best-effort

# Command line interface interactive set up

DATASETS     = ["femnist", "shakespeare"]
AGGREGATORS  = ["fedavg", "balance", "ubar"]
TOPOLOGIES   = ["ring", "fully", "k-regular"]


def _interactive_config() -> dict:
    """Prompt the user for experiment settings when running interactively."""
    import os
    print("\nDecentralised FL Simulator – Interactive Setup")
    print("----------------------------------------------")

    print("\nDataset:")
    for i, d in enumerate(DATASETS, 1):
        print(f"  {i}. {d}")
    ds_idx = int(input(f"Choose dataset (1-{len(DATASETS)}) [default 1]: ").strip() or "1") - 1
    dataset = DATASETS[max(0, min(ds_idx, len(DATASETS) - 1))]

    print("\nAggregation algorithm:")
    for i, a in enumerate(AGGREGATORS, 1):
        print(f"  {i}. {a}")
    ag_idx = int(input("Choose aggregator (1-3) [default 1]: ").strip() or "1") - 1
    aggregation = AGGREGATORS[max(0, min(ag_idx, len(AGGREGATORS) - 1))]

    print("\nTopology:")
    for i, t in enumerate(TOPOLOGIES, 1):
        print(f"  {i}. {t}")
    tp_idx = int(input("Choose topology (1-3) [default 1]: ").strip() or "1") - 1
    topology = TOPOLOGIES[max(0, min(tp_idx, len(TOPOLOGIES) - 1))]

    num_nodes    = int(input("Number of nodes        [default 32]: ").strip() or "32")
    num_rounds   = int(input("Number of rounds       [default 50]: ").strip() or "50")
    attack_ratio = float(input("Attack ratio 0.0-0.5   [default 0.0]:  ").strip() or "0.0")

    from datetime import datetime as _dt
    ts = _dt.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    output = f'results/{dataset}_{aggregation}_{ts}.json'

    print(f"\nConfiguration: {dataset} | {aggregation} | {topology} | "
          f"{num_nodes} nodes | {num_rounds} rounds | attack={attack_ratio}")
    print(f"Output: {output}\n")
    return dict(dataset=dataset, aggregation=aggregation, topology=topology,
                num_nodes=num_nodes, num_rounds=num_rounds,
                attack_ratio=attack_ratio, output=output)


def main():
    parser = argparse.ArgumentParser(description="Decentralised FL Simulator")

    # Basic parameters
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["femnist", "shakespeare"])
    parser.add_argument("--num-nodes", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    
    # Topology
    parser.add_argument("--topology", type=str, default="ring", 
                       choices=["ring", "fully", "k-regular"])
    parser.add_argument("--k", type=int, default=4)
    
    # Aggregation
    parser.add_argument("--aggregation", type=str, default="fedavg",
                       choices=["fedavg", "balance", "ubar"])
    
    # Attack
    parser.add_argument("--attack-ratio", type=float, default=0.0)
    parser.add_argument("--attack-type", type=str, default="directed",
                       choices=["directed", "gaussian"])
    parser.add_argument("--attack-strength", type=float, default=1.0)
    
    # Algorithm parameters
    parser.add_argument("--balance-gamma", type=float, default=2.0)
    parser.add_argument("--balance-kappa", type=float, default=1.0)
    parser.add_argument("--balance-alpha", type=float, default=0.5)
    parser.add_argument("--ubar-rho", type=float, default=0.4)
    
    # Partitioning
    parser.add_argument("--partition-alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID data partitioning (lower = more heterogeneous)")
    
    # Verification layer
    parser.add_argument("--verification", dest="verification", action="store_true", default=True,
                        help="Enable post-acceptance verification layer (default: enabled)")
    parser.add_argument("--no-verification", dest="verification", action="store_false",
                        help="Disable post-acceptance verification layer")
    parser.add_argument("--verification-epsilon", type=float, default=0.05,
                        help="Minimum validation improvement required before flagging/removing a peer")
    parser.add_argument("--trust-decay", type=float, default=0.9,
                        help="Exponential decay for trust score")
    parser.add_argument("--trust-initial", type=float, default=0.5,
                        help="Starting trust for all nodes")
    parser.add_argument("--trust-penalty", type=float, default=0.3,
                        help="Trust reduction on confirmed flag")
    parser.add_argument("--trust-boost", type=float, default=0.05,
                        help="Trust increase on rescue")
    parser.add_argument("--z-low", type=float, default=1.5,
                        help="Lower bound for ambiguous message z-score range")
    parser.add_argument("--z-high", type=float, default=2.5,
                        help="Upper bound for ambiguous message z-score range")
    parser.add_argument("--verification-history-window", type=int, default=10,
                        help="Rounds of history for conservative verification checks")
    parser.add_argument("--rescue-revocation-rounds", type=int, default=3,
                        help="Consecutive anomalous rounds to revoke rescue")

    # Seed (fixed at 42 by default, but can be overridden)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/experiment.json")
    
    args = parser.parse_args()

    # Interactive mode when run directly without
    _interactive = {}
    if args.dataset is None:
        _interactive = _interactive_config()

    dataset     = _interactive.get('dataset',     args.dataset      or 'femnist')
    aggregation = _interactive.get('aggregation', args.aggregation)
    topology    = _interactive.get('topology',    args.topology)
    num_nodes   = _interactive.get('num_nodes',   args.num_nodes)
    num_rounds  = _interactive.get('num_rounds',  args.rounds)
    attack_ratio= _interactive.get('attack_ratio',args.attack_ratio)
    output_path = _interactive.get('output',      args.output)

    # Create config
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
        partition_alpha=args.partition_alpha,
        seed=args.seed,
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
    )
    
    # Run simulation
    simulator = DecentralisedSimulator(config)
    results = simulator.run()

    # Save results
    import os
    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    simulator.save_results(output_path)


if __name__ == "__main__":
    main()
