"""
Generate synthetic LEAF benchmark data for testing the federated learning simulator.
"""
import os
import json
import numpy as np
from pathlib import Path

def create_synthetic_femnist_data(output_dir: str, num_users: int = 10, samples_per_user: int = 100):
    """Create synthetic FEMNIST-like data (28x28 images flattened to 784 features)."""
    
    # Create directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # FEMNIST has 62 classes (10 digits + 26 lowercase + 26 uppercase)
    num_classes = 62
    
    # Create training data
    train_data = {
        'users': [f'user_{i:04d}' for i in range(num_users)],
        'user_data': {}
    }
    
    for user_id in train_data['users']:
        # Random samples and labels
        user_x = np.random.rand(samples_per_user, 784).tolist()  # 28x28 = 784 pixels
        user_y = np.random.randint(0, num_classes, samples_per_user).tolist()
        
        train_data['user_data'][user_id] = {
            'x': user_x,
            'y': user_y
        }
    
    # Save training data
    train_file = os.path.join(train_dir, 'femnist_train.json')
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    print(f"Created {train_file}")
    
    # Create test data (same structure, fewer samples)
    test_data = {
        'users': train_data['users'],
        'user_data': {}
    }
    
    for user_id in test_data['users']:
        test_samples = samples_per_user // 5  # 20% of training data
        user_x = np.random.rand(test_samples, 784).tolist()
        user_y = np.random.randint(0, num_classes, test_samples).tolist()
        
        test_data['user_data'][user_id] = {
            'x': user_x,
            'y': user_y
        }
    
    # Save test data
    test_file = os.path.join(test_dir, 'femnist_test.json')
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    print(f"Created {test_file}")
    
    print(f"\nSynthetic FEMNIST Data Summary:")
    print(f"   Users: {num_users}")
    print(f"   Train samples per user: {samples_per_user}")
    print(f"   Test samples per user: {samples_per_user // 5}")
    print(f"   Classes: {num_classes}")
    print(f"   Image size: 28x28 (784 features)")


def create_synthetic_celeba_data(output_dir: str, num_users: int = 10, samples_per_user: int = 50):
    """Create synthetic CelebA-like data (84x84x3 images)."""
    
    # Create directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # CelebA is a binary classification (gender or age group)
    num_classes = 2
    image_features = 84 * 84 * 3  # 21168 features for flattened image
    
    # Create training data
    train_data = {
        'users': [f'user_{i:04d}' for i in range(num_users)],
        'user_data': {}
    }
    
    for user_id in train_data['users']:
        user_x = np.random.rand(samples_per_user, image_features).tolist()
        user_y = np.random.randint(0, num_classes, samples_per_user).tolist()
        
        train_data['user_data'][user_id] = {
            'x': user_x,
            'y': user_y
        }
    
    # Save training data
    train_file = os.path.join(train_dir, 'celeba_train.json')
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    print(f"Created {train_file}")
    
    # Create test data
    test_data = {
        'users': train_data['users'],
        'user_data': {}
    }
    
    for user_id in test_data['users']:
        test_samples = samples_per_user // 5
        user_x = np.random.rand(test_samples, image_features).tolist()
        user_y = np.random.randint(0, num_classes, test_samples).tolist()
        
        test_data['user_data'][user_id] = {
            'x': user_x,
            'y': user_y
        }
    
    # Save test data
    test_file = os.path.join(test_dir, 'celeba_test.json')
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    print(f"Created {test_file}")
    
    print(f"\nSynthetic CelebA Data Summary:")
    print(f"   Users: {num_users}")
    print(f"   Train samples per user: {samples_per_user}")
    print(f"   Test samples per user: {samples_per_user // 5}")
    print(f"   Classes: {num_classes}")
    print(f"   Image size: 84x84x3 (21168 features)")


def create_synthetic_shakespeare_data(
    output_dir: str,
    num_users: int = 10,
    samples_per_user: int = 100,
):
    """Create synthetic Shakespeare-like data.

    Each sample x is a sequence of 80 character IDs (ints in [0, 79])
    and y is the next character ID.  This matches the character-level
    LSTM inputs expected by LEAFShakespeareModel.
    """
    train_dir = os.path.join(output_dir, 'train')
    test_dir  = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    np.random.seed(42)
    vocab_size = 80
    seq_len    = 80

    def _build_split(n_samples):
        data = {'users': [f'user_{i:04d}' for i in range(num_users)],
                'user_data': {}}
        for uid in data['users']:
            xs = np.random.randint(0, vocab_size,
                                   size=(n_samples, seq_len)).tolist()
            ys = np.random.randint(0, vocab_size, size=n_samples).tolist()
            data['user_data'][uid] = {'x': xs, 'y': ys}
        return data

    train_file = os.path.join(train_dir, 'shakespeare_train.json')
    test_file  = os.path.join(test_dir,  'shakespeare_test.json')
    with open(train_file, 'w') as f:
        json.dump(_build_split(samples_per_user), f)
    with open(test_file, 'w') as f:
        json.dump(_build_split(max(1, samples_per_user // 5)), f)

    print(f"Created {train_file}")
    print(f"Created {test_file}")
    print(f"\nSynthetic Shakespeare Data Summary:")
    print(f"   Users: {num_users}")
    print(f"   Train samples/user: {samples_per_user}")
    print(f"   Test samples/user:  {samples_per_user // 5}")
    print(f"   Vocab size: {vocab_size}  |  Sequence length: {seq_len}")


def create_synthetic_reddit_data(
    output_dir: str,
    num_users: int = 10,
    samples_per_user: int = 100,
):
    """Create synthetic Reddit-like data.

    Each sample x is a sequence of 10 word IDs (ints in [0, 999])
    and y is the next word ID.  This matches the word-level LSTM inputs
    expected by LEAFRedditModel.
    """
    train_dir = os.path.join(output_dir, 'train')
    test_dir  = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    np.random.seed(42)
    vocab_size = 1000
    seq_len    = 10

    def _build_split(n_samples):
        data = {'users': [f'user_{i:04d}' for i in range(num_users)],
                'user_data': {}}
        for uid in data['users']:
            xs = np.random.randint(0, vocab_size,
                                   size=(n_samples, seq_len)).tolist()
            ys = np.random.randint(0, vocab_size, size=n_samples).tolist()
            data['user_data'][uid] = {'x': xs, 'y': ys}
        return data

    train_file = os.path.join(train_dir, 'reddit_train.json')
    test_file  = os.path.join(test_dir,  'reddit_test.json')
    with open(train_file, 'w') as f:
        json.dump(_build_split(samples_per_user), f)
    with open(test_file, 'w') as f:
        json.dump(_build_split(max(1, samples_per_user // 5)), f)

    print(f"Created {train_file}")
    print(f"Created {test_file}")
    print(f"\nSynthetic Reddit Data Summary:")
    print(f"   Users: {num_users}")
    print(f"   Train samples/user: {samples_per_user}")
    print(f"   Test samples/user:  {samples_per_user // 5}")
    print(f"   Vocab size: {vocab_size}  |  Sequence length: {seq_len}")


if __name__ == "__main__":
    base_path = "leaf/data"

    print("Setting up synthetic LEAF benchmark data...\n")

    # FEMNIST
    create_synthetic_femnist_data(
        os.path.join(base_path, "femnist", "data"),
        num_users=10, samples_per_user=100,
    )

    # CelebA
    create_synthetic_celeba_data(
        os.path.join(base_path, "celeba", "data"),
        num_users=10, samples_per_user=50,
    )

    # Shakespeare
    create_synthetic_shakespeare_data(
        os.path.join(base_path, "shakespeare", "data"),
        num_users=10, samples_per_user=100,
    )

    # Reddit
    create_synthetic_reddit_data(
        os.path.join(base_path, "reddit", "data"),
        num_users=10, samples_per_user=100,
    )

    print("\nSynthetic data setup complete! You can now run your simulator.")
