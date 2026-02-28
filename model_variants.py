# model_variants.py
import torch.nn as nn

def get_model_variant(variant: str, num_classes: int):
    """Return a FEMNIST CNN whose capacity matches the requested variant name."""
    
    if variant == "tiny":
        return TinyFEMNISTModel(num_classes)
    elif variant == "small":
        return SmallFEMNISTModel(num_classes)
    elif variant == "baseline":
        from leaf_datasets import LEAFFEMNISTModel
        return LEAFFEMNISTModel(num_classes)
    elif variant == "large":
        return LargeFEMNISTModel(num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")

class TinyFEMNISTModel(nn.Module):
    """Minimal single-conv FEMNIST model (~80K params) for quick smoke tests."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class SmallFEMNISTModel(nn.Module):
    """Single-conv FEMNIST model (~250K params) for lightweight experiments."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class LargeFEMNISTModel(nn.Module):
    """Two-conv FEMNIST model (~1M params) with dropout for full-scale experiments."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 128 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)