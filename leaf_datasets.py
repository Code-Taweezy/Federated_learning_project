import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class LEAFDataset(Dataset):
    """Dataset loader for LEAF benchmark datasets."""
    
    def __init__(self, data_dir: str, train: bool = True):
        self.data_dir = data_dir
        self.train = train
        self.data = []
        self.targets = []
        
        # Load all JSON files
        split_dir = os.path.join(data_dir, 'train' if train else 'test')
        
        for filename in os.listdir(split_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(split_dir, filename)
                with open(filepath, 'r') as f:
                    file_data = json.load(f)
                    
                for user in file_data['users']:
                    user_data = file_data['user_data'][user]
                    self.data.extend(user_data['x'])
                    self.targets.extend(user_data['y'])
        
        print(f"Loaded {len(self.data)} samples from {split_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = self.targets[idx]
        # FEMNIST 28x28 grayscale
        if len(x.shape) == 1 and x.shape[0] == 784:
            x = x.reshape(1, 28, 28)
        # CelebA 84x84 RGB
        elif len(x.shape) == 1 and x.shape[0] == 84 * 84 * 3:
            x = x.reshape(3, 84, 84)
        return x, y

class LEAFFEMNISTModel(nn.Module):
    """CNN model for FEMNIST dataset."""
    
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class LEAFCelebAModel(nn.Module):
    """CNN model for CelebA dataset."""
    
    def __init__(self, num_classes=2, image_size=(84, 84)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        size = image_size[0] // 8  # 3 pooling layers
        self.fc1 = nn.Linear(128 * size * size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class LEAFTextDataset(Dataset):
    """Dataset for character- or word-level sequence LEAF datasets
    (Shakespeare, Reddit).  Each sample x is a sequence of integer IDs
    and y is the target class integer.
    """

    def __init__(self, data_dir: str, train: bool = True):
        self.data = []
        self.targets = []
        split_dir = os.path.join(data_dir, 'train' if train else 'test')
        for filename in os.listdir(split_dir):
            if filename.endswith('.json'):
                with open(os.path.join(split_dir, filename), 'r') as f:
                    file_data = json.load(f)
                for user in file_data['users']:
                    ud = file_data['user_data'][user]
                    self.data.extend(ud['x'])
                    self.targets.extend(ud['y'])
        print(f"Loaded {len(self.data)} samples from {split_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx])
        y = int(self.targets[idx])
        return x, y


class LEAFShakespeareModel(nn.Module):
    """Character-level 2-layer LSTM for Shakespeare next-character prediction.
    Vocabulary: 80 printable ASCII characters.
    """
    VOCAB_SIZE = 80

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, 8)
        self.lstm = nn.LSTM(8, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.long()
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out[:, -1, :])


class LEAFRedditModel(nn.Module):
    """Word-level 2-layer LSTM for Reddit next-word prediction.
    Vocabulary: 1000 most-common words (synthetic).
    """
    VOCAB_SIZE = 1000

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, 256)
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.long()
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out[:, -1, :])


def load_leaf_dataset(dataset_name: str, data_path: str):
    """Load LEAF dataset."""
    if dataset_name.lower() == 'femnist':
        train_ds = LEAFDataset(data_path, train=True)
        test_ds = LEAFDataset(data_path, train=False)
        model = LEAFFEMNISTModel(num_classes=62)
        return train_ds, test_ds, model, 62, (28, 28)
    elif dataset_name.lower() == 'celeba':
        train_ds = LEAFDataset(data_path, train=True)
        test_ds = LEAFDataset(data_path, train=False)
        model = LEAFCelebAModel(num_classes=2, image_size=(84, 84))
        return train_ds, test_ds, model, 2, (84, 84)
    elif dataset_name.lower() == 'shakespeare':
        train_ds = LEAFTextDataset(data_path, train=True)
        test_ds = LEAFTextDataset(data_path, train=False)
        model = LEAFShakespeareModel(num_classes=80)
        return train_ds, test_ds, model, 80, (80,)
    elif dataset_name.lower() == 'reddit':
        train_ds = LEAFTextDataset(data_path, train=True)
        test_ds = LEAFTextDataset(data_path, train=False)
        model = LEAFRedditModel(num_classes=1000)
        return train_ds, test_ds, model, 1000, (10,)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def _dirichlet_partition(targets, num_clients: int, alpha: float, rng: np.random.RandomState):
    """Partition indices using a Dirichlet distribution for non-IID label skew.

    For each class, samples proportions from Dir(alpha) over clients and
    allocates the corresponding fraction of that class's indices to each client.
    Every client is guaranteed at least one sample.
    """
    targets = np.array(targets)
    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        # Draw Dirichlet proportions and compute split points
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[client_id].extend(chunk.tolist())

    # Guarantee every client has at least one sample
    for client_id in range(num_clients):
        if len(client_indices[client_id]) == 0:
            # Steal one sample from the largest client
            donor = max(range(num_clients), key=lambda i: len(client_indices[i]))
            client_indices[client_id].append(client_indices[donor].pop())

    return [np.array(idx) for idx in client_indices]


def create_leaf_client_partitions(train_ds, test_ds, num_clients: int, seed: int = 42,
                                  alpha: float = 0.5):
    """Partition data across clients.

    When *alpha* is provided (default 0.5), a Dirichlet-based non-IID split is
    used so that each client receives a skewed class distribution.  Higher alpha
    values approach IID behaviour; lower values increase heterogeneity.
    """
    rng = np.random.RandomState(seed)

    train_targets = np.array(train_ds.targets)
    test_targets  = np.array(test_ds.targets)

    train_partitions = _dirichlet_partition(train_targets, num_clients, alpha, rng)
    test_partitions  = _dirichlet_partition(test_targets,  num_clients, alpha, rng)

    # Print a brief heterogeneity summary
    classes_per_client = [len(np.unique(train_targets[part])) for part in train_partitions]
    print(f"Partitioned data (non-IID, alpha={alpha})")
    print(f"   Client class counts: min={min(classes_per_client)}, "
          f"max={max(classes_per_client)} classes per client")

    return train_partitions, test_partitions