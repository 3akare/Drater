import json
import logging
import numpy as np

def save_config(config, filepath):
    """Saves the configuration to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Configuration saved to {filepath}")

def load_config(filepath):
    """Loads configuration from a JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    logging.info(f"Configuration loaded from {filepath}")
    return config

def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int, padding_value: float = 0.0) -> np.ndarray:
    """Pads or truncates a NumPy sequence to a target length."""
    if sequence.shape[0] > target_length:
        return sequence[:target_length, :]
    elif sequence.shape[0] < target_length:
        padding = np.full((target_length - sequence.shape[0], sequence.shape[1]), padding_value, dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)
    return sequence

def add_noise(sequence: np.ndarray, noise_level=0.001) -> np.ndarray:
    """Adds Gaussian noise to a keypoint sequence."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_sequence(sequence: np.ndarray, scale_range=(0.8, 1.2)) -> np.ndarray:
    """Scales a keypoint sequence."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return sequence * scale_factor

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """Applies a random set of augmentations to a sequence."""
    augmented_sequence = sequence.copy()
    if np.random.rand() < 0.8:
        augmented_sequence = add_noise(augmented_sequence, noise_level=0.05)
    if np.random.rand() < 0.8:
        augmented_sequence = scale_sequence(augmented_sequence, scale_range=(0.7, 1.3))
    return sequence

def data_generator(X_data, y_data, batch_size, augment=False):
    """Generates batches of data with optional on-the-fly augmentation."""
    num_samples = len(X_data)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch, y_batch = X_data[batch_indices], y_data[batch_indices]
            if augment:
                X_batch_augmented = np.array([augment_sequence(seq) for seq in X_batch])
                yield X_batch_augmented, y_batch
            else:
                yield X_batch, y_batch
