"""
Data preprocessing module for multi-output side-channel attack.

Implements:
- Multi-output labeling (LSB formula)
- Dataset reconstruction
- Noise injection
- De-synchronization simulation
"""

import numpy as np
from typing import Tuple, Optional
import os


# AES S-box lookup table
SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)


def aes_sbox(byte: int) -> int:
    """Apply AES S-box transformation."""
    return SBOX[byte]


def compute_lsb_label(plaintext: int, key_guess: int) -> int:
    """
    Compute LSB label for multi-output classification.
    
    Formula: l_i^j = LSB(Sbox(p_i ⊕ k_j))
    
    Args:
        plaintext: Plaintext byte (0-255)
        key_guess: Key guess byte (0-255)
        
    Returns:
        LSB of S-box output (0 or 1)
    """
    intermediate = plaintext ^ key_guess
    sbox_output = aes_sbox(intermediate)
    return sbox_output & 1  # LSB


def create_multi_output_labels(plaintexts: np.ndarray) -> np.ndarray:
    """
    Create multi-output labels for all 256 key hypotheses.
    
    Args:
        plaintexts: Array of plaintext bytes (n_traces,)
        
    Returns:
        Labels array of shape (n_traces, 256) where each column
        corresponds to a key hypothesis (0-255)
    """
    n_traces = len(plaintexts)
    labels = np.zeros((n_traces, 256), dtype=np.int64)
    
    for i, plaintext in enumerate(plaintexts):
        for key_guess in range(256):
            labels[i, key_guess] = compute_lsb_label(plaintext, key_guess)
    
    return labels


def inject_gaussian_noise(traces: np.ndarray, sigma: float, mean: float = 0.0) -> np.ndarray:
    """
    Inject Gaussian noise into power traces.
    
    Formula: t_noise(i,m) = t(i,m) + σ × randn(1,m) + mean
    
    Args:
        traces: Power traces array (n_traces, trace_length)
        sigma: Standard deviation of noise (σ)
        mean: Mean of noise (default: 0.0)
        
    Returns:
        Noisy traces with same shape as input
    """
    noise = np.random.normal(mean, sigma, traces.shape)
    return traces + noise


def apply_desynchronization(traces: np.ndarray, max_shift: int = 20) -> np.ndarray:
    """
    Apply de-synchronization by randomly shifting traces.
    
    Args:
        traces: Power traces array (n_traces, trace_length)
        max_shift: Maximum number of samples to shift (default: 20)
        
    Returns:
        De-synchronized traces (may have different length if padding needed)
    """
    n_traces, trace_length = traces.shape
    desync_traces = []
    
    for i in range(n_traces):
        shift = np.random.randint(0, max_shift + 1)
        trace = traces[i]
        
        if shift > 0:
            # Shift right by padding zeros at the beginning
            shifted = np.pad(trace, (shift, 0), mode='constant')[:trace_length]
        else:
            shifted = trace
        
        desync_traces.append(shifted)
    
    return np.array(desync_traces)


def normalize_traces(traces: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    """
    Normalize power traces.
    
    Args:
        traces: Power traces array (n_traces, trace_length)
        method: Normalization method ('standard', 'minmax', 'l2')
        
    Returns:
        Normalized traces and normalization parameters
    """
    if method == 'standard':
        mean = np.mean(traces, axis=0, keepdims=True)
        std = np.std(traces, axis=0, keepdims=True)
        # Ensure std is a regular numeric array (not structured/void) before comparison
        std = np.asarray(std, dtype=np.float64)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        normalized = (traces - mean) / std
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(traces, axis=0, keepdims=True)
        max_val = np.max(traces, axis=0, keepdims=True)
        range_val = max_val - min_val
        # Ensure range_val is a regular numeric array (not structured/void) before comparison
        range_val = np.asarray(range_val, dtype=np.float64)
        range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
        normalized = (traces - min_val) / range_val
        params = {'min': min_val, 'max': max_val}
        
    elif method == 'l2':
        norms = np.linalg.norm(traces, axis=1, keepdims=True)
        # Ensure norms is a regular numeric array (not structured/void) before comparison
        norms = np.asarray(norms, dtype=np.float64)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = traces / norms
        params = {'norms': norms}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def load_ascad_dataset(dataset_path: str, dataset_name: str = "Dataset2", 
                      max_traces: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Load ASCAD dataset.
    
    Args:
        dataset_path: Path to ASCAD HDF5 file or directory containing .h5 file
        dataset_name: Name of dataset (Dataset1, Dataset2, Dataset3) - used if dataset_path is directory
        max_traces: Optional maximum number of traces to load (for testing)
        
    Returns:
        Tuple of (traces, plaintexts, correct_key)
        correct_key may be None if not available
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for ASCAD dataset. Install with: pip install h5py")
    
    # Handle both file path and directory path
    if os.path.isfile(dataset_path):
        hdf5_path = dataset_path
    else:
        hdf5_path = os.path.join(dataset_path, f"{dataset_name}.h5")
    
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Dataset not found: {hdf5_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # ASCAD v2 format: 'traces' and 'metadata' groups
            if 'traces' in f:
                traces = f['traces'][:]
            elif 'Profiling_traces' in f:
                traces = f['Profiling_traces']['traces'][:]
            else:
                # Try to find traces in any group
                traces_key = None
                for key in f.keys():
                    if 'trace' in key.lower():
                        if isinstance(f[key], h5py.Dataset):
                            traces_key = key
                            break
                        elif isinstance(f[key], h5py.Group) and 'traces' in f[key]:
                            traces = f[key]['traces'][:]
                            traces_key = 'found'
                            break
                if traces_key is None:
                    raise KeyError("Could not find traces in HDF5 file. Available keys: " + str(list(f.keys())))
                if traces_key != 'found':
                    traces = f[traces_key][:]
            
            # Get metadata (plaintexts, keys, etc.)
            if 'metadata' in f:
                metadata = f['metadata']
            elif 'Profiling_traces' in f and 'metadata' in f['Profiling_traces']:
                metadata = f['Profiling_traces']['metadata']
            else:
                metadata = None
            
            if metadata is not None:
                # Get plaintexts
                if 'plaintext' in metadata:
                    plaintexts = metadata['plaintext'][:]
                elif 'input' in metadata:
                    plaintexts = metadata['input'][:]
                else:
                    raise KeyError("Could not find plaintext in metadata. Available keys: " + str(list(metadata.keys())))
                
                # Ensure plaintexts is a regular numpy array (not structured/void)
                plaintexts = np.asarray(plaintexts, dtype=np.uint8)
                
                # Get correct key (usually first byte of key)
                correct_key = None
                if 'key' in metadata:
                    key_data = metadata['key'][:]
                    # Convert to numpy array and ensure proper dtype
                    key_data = np.asarray(key_data)
                    if len(key_data.shape) > 1:
                        # Key is array of bytes, take first byte
                        key_byte = key_data[0, 0] if key_data.shape[1] > 0 else key_data[0]
                    else:
                        key_byte = key_data[0]
                    # Convert to plain Python int to avoid structured/void dtype issues
                    correct_key = int(np.asarray(key_byte, dtype=np.uint8).item())
            else:
                # If no metadata, create dummy plaintexts (will need to be provided separately)
                print("Warning: No metadata found. Creating dummy plaintexts.")
                plaintexts = np.zeros((len(traces), 16), dtype=np.uint8)
                correct_key = None
            
            # Limit traces if requested
            if max_traces is not None and len(traces) > max_traces:
                traces = traces[:max_traces]
                plaintexts = plaintexts[:max_traces]
            
            # Extract first byte of plaintext for single-byte attack
            # ASCAD typically uses the first byte for single-byte attacks
            if len(plaintexts.shape) > 1:
                plaintexts = plaintexts[:, 0]  # Take first byte
            
            return traces, plaintexts, correct_key
        
    except Exception as e:
        raise RuntimeError(f"Error loading ASCAD dataset from {hdf5_path}: {e}")


def load_chipwhisperer_dataset(dataset_path: str, dataset_name: str = "Dataset4") -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Load ChipWhisperer dataset.
    
    Args:
        dataset_path: Path to ChipWhisperer dataset directory
        dataset_name: Name of dataset (Dataset4, Dataset5)
        
    Returns:
        Tuple of (traces, plaintexts, correct_key)
        correct_key may be None if not available
    """
    # This is a placeholder - actual implementation depends on ChipWhisperer format
    # ChipWhisperer may use numpy arrays or custom format
    npz_path = os.path.join(dataset_path, f"{dataset_name}.npz")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    
    data = np.load(npz_path)
    traces = data['traces']
    plaintexts = data['plaintexts']
    correct_key = data.get('correct_key', None)
    
    return traces, plaintexts, correct_key


def create_noisy_dataset(traces: np.ndarray, plaintexts: np.ndarray, 
                        sigma: float, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create noisy variant of dataset.
    
    Args:
        traces: Original traces
        plaintexts: Original plaintexts
        sigma: Noise standard deviation
        dataset_name: Name for the noisy dataset
        
    Returns:
        Tuple of (noisy_traces, plaintexts)
    """
    noisy_traces = inject_gaussian_noise(traces, sigma=sigma)
    return noisy_traces, plaintexts


def create_desync_dataset(traces: np.ndarray, plaintexts: np.ndarray,
                         max_shift: int = 20, dataset_name: str = "sh20") -> Tuple[np.ndarray, np.ndarray]:
    """
    Create de-synchronized variant of dataset.
    
    Args:
        traces: Original traces
        plaintexts: Original plaintexts
        max_shift: Maximum shift in samples
        dataset_name: Name suffix for the dataset
        
    Returns:
        Tuple of (desync_traces, plaintexts)
    """
    desync_traces = apply_desynchronization(traces, max_shift=max_shift)
    return desync_traces, plaintexts

