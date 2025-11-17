"""
Dataset Loading and Preprocessing for Side-Channel Attacks

This module handles loading ASCAD datasets and applying countermeasures
(noise generation, de-synchronization).
"""

import numpy as np
import os
from pathlib import Path

# h5py is imported conditionally in load_ascad_dataset to provide better error messages
# Try to import h5py, but don't fail if it's not available
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def load_ascad_dataset(dataset_path, group='Profiling_traces'):
    """
    Load ASCAD dataset from HDF5 file.
    
    Args:
        dataset_path: Path to ASCAD HDF5 file
        group: HDF5 group name ('Profiling_traces' or 'Attack_traces')
        
    Returns:
        traces: Power traces array (N, trace_length)
        plaintexts: Plaintext bytes array (N, 16) for AES-128
        keys: Key bytes array (N, 16) for AES-128
        metadata: Dictionary with additional metadata
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required to load ASCAD datasets. Install with: pip install h5py")
    
    with h5py.File(dataset_path, 'r') as f:
        traces = np.array(f[group]['traces'])
        plaintexts = np.array(f[group]['plaintext'])
        keys = np.array(f[group]['key'])
        
        metadata = {
            'trace_length': traces.shape[1],
            'num_traces': traces.shape[0],
            'group': group
        }
        
        # Try to get metadata if available
        if 'metadata' in f[group]:
            metadata.update(dict(f[group]['metadata'].attrs))
    
    return traces, plaintexts, keys, metadata


def add_gaussian_noise(traces, sigma=1.0, mean=0.0, seed=None):
    """
    Add Gaussian noise to power traces to simulate noise-generation countermeasure.
    
    Formula: t_noise(i, m) = t(i, m) + σ × randn(1, m) + mean
    
    Args:
        traces: Power traces array (N, trace_length)
        sigma: Standard deviation of noise (default 1.0)
        mean: Mean of noise (default 0.0)
        seed: Random seed for reproducibility
        
    Returns:
        Noisy traces array (N, trace_length)
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(mean, sigma, traces.shape)
    noisy_traces = traces + noise
    
    return noisy_traces


def apply_desynchronization(traces, max_shift=20, seed=None):
    """
    Apply de-synchronization countermeasure by randomly shifting traces.
    
    Args:
        traces: Power traces array (N, trace_length)
        max_shift: Maximum number of samples to shift (default 20)
        seed: Random seed for reproducibility
        
    Returns:
        Desynchronized traces array (N, trace_length)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_traces, trace_length = traces.shape
    desync_traces = np.zeros_like(traces)
    
    for i in range(n_traces):
        # Random shift for each trace
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift == 0:
            desync_traces[i] = traces[i]
        elif shift > 0:
            # Shift right (pad left with zeros)
            desync_traces[i, shift:] = traces[i, :-shift]
        else:
            # Shift left (pad right with zeros)
            shift = abs(shift)
            desync_traces[i, :-shift] = traces[i, shift:]
    
    return desync_traces


def create_dataset_variants(traces, plaintexts, keys, base_name='Dataset'):
    """
    Create different dataset variants with varying trace counts.
    
    Args:
        traces: Power traces array
        plaintexts: Plaintext bytes array
        keys: Key bytes array
        base_name: Base name for dataset variants
        
    Returns:
        Dictionary of dataset variants
    """
    n_total = len(traces)
    
    datasets = {}
    
    # Dataset1: ~10,000 traces
    n1 = min(10000, n_total)
    datasets[f'{base_name}1'] = {
        'traces': traces[:n1],
        'plaintexts': plaintexts[:n1],
        'keys': keys[:n1]
    }
    
    # Dataset2: ~20,000 traces
    n2 = min(20000, n_total)
    if n2 > n1:
        datasets[f'{base_name}2'] = {
            'traces': traces[:n2],
            'plaintexts': plaintexts[:n2],
            'keys': keys[:n2]
        }
    
    # Dataset3: ~50,000 traces
    n3 = min(50000, n_total)
    if n3 > n2:
        datasets[f'{base_name}3'] = {
            'traces': traces[:n3],
            'plaintexts': plaintexts[:n3],
            'keys': keys[:n3]
        }
    
    return datasets


def create_noisy_datasets(datasets, noise_levels=[0.5, 1.0, 1.5], seed=None):
    """
    Create noisy variants of datasets.
    
    Args:
        datasets: Dictionary of datasets
        noise_levels: List of sigma values for noise
        seed: Random seed
        
    Returns:
        Dictionary with noisy dataset variants (e.g., Dataset1-N1, Dataset1-N2, etc.)
    """
    noisy_datasets = {}
    
    for dataset_name, data in datasets.items():
        traces = data['traces']
        
        for i, sigma in enumerate(noise_levels, start=1):
            noisy_traces = add_gaussian_noise(traces, sigma=sigma, seed=seed)
            noisy_datasets[f'{dataset_name}-N{i}'] = {
                'traces': noisy_traces,
                'plaintexts': data['plaintexts'],
                'keys': data['keys']
            }
    
    return noisy_datasets


def create_desync_datasets(datasets, max_shift=20, seed=None):
    """
    Create de-synchronized variants of datasets.
    
    Args:
        datasets: Dictionary of datasets
        max_shift: Maximum shift amount
        seed: Random seed
        
    Returns:
        Dictionary with desynchronized dataset variants
    """
    desync_datasets = {}
    
    for dataset_name, data in datasets.items():
        traces = data['traces']
        desync_traces = apply_desynchronization(traces, max_shift=max_shift, seed=seed)
        desync_datasets[f'{dataset_name}-sh{max_shift}'] = {
            'traces': desync_traces,
            'plaintexts': data['plaintexts'],
            'keys': data['keys']
        }
    
    return desync_datasets

