
import numpy as np

def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int, padding_value: float = 0.0) -> np.ndarray:
    """
    Pads or truncates a NumPy sequence to a target length.

    Args:
        sequence (np.ndarray): The input NumPy array representing a sequence of keypoints.
                               Shape: (num_frames, feature_dim).
        target_length (int): The desired length of the sequence.
        padding_value (float): The value to use for padding.

    Returns:
        np.ndarray: The padded or truncated sequence.
    """
    if sequence.shape[0] > target_length:
        return sequence[:target_length, :]
    elif sequence.shape[0] < target_length:
        padding = np.full((target_length - sequence.shape[0], sequence.shape[1]), padding_value, dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)
    return sequence