"""
Main experiment scripts for reproducing paper results.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import (
    create_multi_output_labels,
    normalize_traces,
    load_ascad_dataset,
    load_chipwhisperer_dataset,
    create_noisy_dataset,
    create_desync_dataset
)
from mlp_mo import MLP_MO
from cnn_mo import CNN_MO
from training import PowerTraceDataset, TrainingLogger, train_mlp_mo, train_cnn_mo
from evaluation import evaluate_model, run_repeated_attacks, compute_success_rate
from visualization import (
    plot_accuracy_curves,
    plot_attack_time_comparison,
    plot_success_rate_comparison,
    plot_branch_accuracies
)


def experiment_masking_mlp_mo(dataset_path: str, dataset_name: str = "Dataset2",
                              shared_layer_size: int = 200, num_epochs: int = 50,
                              batch_size: int = 32, device: str = 'cpu'):
    """
    Experiment 1: Train MLP_MO on masking countermeasure dataset.
    
    Args:
        dataset_path: Path to ASCAD dataset
        dataset_name: Name of dataset (Dataset1, Dataset2, Dataset3)
        shared_layer_size: Size of shared layer (0 for Non-SoSL, 200 for SoSL-200)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
    """
    print(f"\n{'='*60}")
    print(f"Experiment: MLP_MO on Masking Countermeasure")
    print(f"Dataset: {dataset_name}, Shared Layer: {shared_layer_size}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    traces, plaintexts, correct_key = load_ascad_dataset(dataset_path, dataset_name)
    print(f"Loaded {len(traces)} traces, trace length: {traces.shape[1]}")
    if correct_key is not None:
        print(f"Correct key: {correct_key}")
    
    # Normalize traces
    traces, norm_params = normalize_traces(traces, method='standard')
    
    # Create multi-output labels
    print("Creating multi-output labels...")
    labels = create_multi_output_labels(plaintexts)
    print(f"Labels shape: {labels.shape}")
    
    # Split dataset (80% train, 20% val)
    n_train = int(0.8 * len(traces))
    train_traces = traces[:n_train]
    train_labels = labels[:n_train]
    val_traces = traces[n_train:]
    val_labels = labels[n_train:]
    
    # Create datasets and loaders
    train_dataset = PowerTraceDataset(train_traces, train_labels)
    val_dataset = PowerTraceDataset(val_traces, val_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    trace_length = traces.shape[1]
    model = MLP_MO(trace_length=trace_length, shared_layer_size=shared_layer_size)
    print(f"\nModel created:")
    print(f"  Trace length: {trace_length}")
    print(f"  Shared layer size: {shared_layer_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create logger
    logger = TrainingLogger()
    
    # Train model
    print("\nStarting training...")
    history = train_mlp_mo(model, train_loader, val_loader, num_epochs=num_epochs,
                          device=device, logger=logger, correct_key=correct_key)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader, device=device, correct_key=correct_key)
    
    print(f"\nResults:")
    print(f"  Attack time: {history['attack_time']:.2f} seconds")
    print(f"  Mean accuracy: {metrics['mean_accuracy']:.4f}")
    if correct_key is not None:
        print(f"  Correct key accuracy: {metrics['correct_key_accuracy']:.4f}")
        print(f"  Accuracy gap: {metrics['accuracy_gap']:.4f}")
        print(f"  Key ranking: {metrics['key_ranking']}")
        print(f"  Success: {metrics['success']}")
    
    # Save results
    model_name = f"MLP_MO_SoSL{shared_layer_size}" if shared_layer_size > 0 else "MLP_MO_NonSoSL"
    logger.save(f"{model_name}_{dataset_name}_training.json")
    
    # Plot results
    plot_accuracy_curves(history, 
                        save_path=f"figures/masking_{model_name}_{dataset_name}_accuracy.png",
                        title=f"{model_name} on {dataset_name}")
    
    if correct_key is not None:
        plot_branch_accuracies(metrics['branch_accuracies'], correct_key=correct_key,
                              save_path=f"figures/masking_{model_name}_{dataset_name}_branches.png",
                              title=f"Branch Accuracies - {model_name} on {dataset_name}")
    
    return history, metrics


def experiment_noise_robustness(dataset_path: str, dataset_name: str = "Dataset2",
                                sigmas: list = [0.5, 1.0, 1.5], num_attacks: int = 50,
                                num_epochs: int = 50, batch_size: int = 32, device: str = 'cpu'):
    """
    Experiment 2: Evaluate noise robustness.
    
    Args:
        dataset_path: Path to ASCAD dataset
        dataset_name: Name of base dataset
        sigmas: List of noise levels to test
        num_attacks: Number of repeated attacks per noise level
        num_epochs: Number of epochs per attack
        batch_size: Batch size
        device: Device to train on
    """
    print(f"\n{'='*60}")
    print(f"Experiment: Noise Robustness")
    print(f"Dataset: {dataset_name}, Noise levels: {sigmas}")
    print(f"{'='*60}\n")
    
    # Load base dataset
    traces, plaintexts, correct_key = load_ascad_dataset(dataset_path, dataset_name)
    traces, _ = normalize_traces(traces, method='standard')
    labels = create_multi_output_labels(plaintexts)
    
    if correct_key is None:
        print("Warning: Correct key not available. Cannot compute success rate.")
        return
    
    results = {}
    
    for sigma in sigmas:
        print(f"\n{'='*40}")
        print(f"Testing noise level σ = {sigma}")
        print(f"{'='*40}\n")
        
        # Create noisy dataset
        noisy_traces, _ = create_noisy_dataset(traces, plaintexts, sigma=sigma, 
                                              dataset_name=f"{dataset_name}-N{sigma}")
        noisy_traces, _ = normalize_traces(noisy_traces, method='standard')
        
        # Split dataset
        n_train = int(0.8 * len(noisy_traces))
        train_traces = noisy_traces[:n_train]
        train_labels = labels[:n_train]
        val_traces = noisy_traces[n_train:]
        val_labels = labels[n_train:]
        
        train_dataset = PowerTraceDataset(train_traces, train_labels)
        val_dataset = PowerTraceDataset(val_traces, val_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Run repeated attacks
        trace_length = traces.shape[1]
        model_kwargs = {'trace_length': trace_length, 'shared_layer_size': 200}  # SoSL-200
        
        repeated_results = run_repeated_attacks(
            MLP_MO, model_kwargs, train_loader, val_loader,
            num_attacks=num_attacks, num_epochs=num_epochs,
            correct_key=correct_key, device=device
        )
        
        results[f'σ={sigma}'] = repeated_results['success_rate']
        
        print(f"\nResults for σ = {sigma}:")
        print(f"  Success rate: {repeated_results['success_rate']*100:.1f}%")
        print(f"  Mean attack time: {repeated_results['mean_attack_time']:.2f} seconds")
    
    # Plot results
    plot_success_rate_comparison(results,
                                save_path=f"figures/noise_success_rate_{dataset_name}.png",
                                title=f"Success Rate vs Noise Level - {dataset_name}")
    
    return results


def experiment_desync_cnn_mo(dataset_path: str, dataset_name: str = "Dataset4",
                             max_shift: int = 20, num_epochs: int = 50,
                             batch_size: int = 32, device: str = 'cpu'):
    """
    Experiment 3: Train CNN_MO on de-synchronized data.
    
    Args:
        dataset_path: Path to ChipWhisperer dataset
        dataset_name: Name of dataset (Dataset4, Dataset5)
        max_shift: Maximum shift for de-synchronization
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
    """
    print(f"\n{'='*60}")
    print(f"Experiment: CNN_MO on De-Synchronization")
    print(f"Dataset: {dataset_name}, Max shift: {max_shift}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    traces, plaintexts, correct_key = load_chipwhisperer_dataset(dataset_path, dataset_name)
    print(f"Loaded {len(traces)} traces, trace length: {traces.shape[1]}")
    if correct_key is not None:
        print(f"Correct key: {correct_key}")
    
    # Normalize traces
    traces, _ = normalize_traces(traces, method='standard')
    
    # Apply de-synchronization
    print(f"Applying de-synchronization (max shift: {max_shift})...")
    desync_traces, _ = create_desync_dataset(traces, plaintexts, max_shift=max_shift)
    desync_traces, _ = normalize_traces(desync_traces, method='standard')
    
    # Create multi-output labels
    print("Creating multi-output labels...")
    labels = create_multi_output_labels(plaintexts)
    
    # Split dataset
    n_train = int(0.8 * len(desync_traces))
    train_traces = desync_traces[:n_train]
    train_labels = labels[:n_train]
    val_traces = desync_traces[n_train:]
    val_labels = labels[n_train:]
    
    # Create datasets and loaders
    train_dataset = PowerTraceDataset(train_traces, train_labels)
    val_dataset = PowerTraceDataset(val_traces, val_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    trace_length = traces.shape[1]
    model = CNN_MO(trace_length=trace_length)
    print(f"\nModel created:")
    print(f"  Trace length: {trace_length}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create logger
    logger = TrainingLogger()
    
    # Train model
    print("\nStarting training...")
    history = train_cnn_mo(model, train_loader, val_loader, num_epochs=num_epochs,
                          device=device, logger=logger, correct_key=correct_key)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader, device=device, correct_key=correct_key)
    
    print(f"\nResults:")
    print(f"  Attack time: {history['attack_time']:.2f} seconds")
    print(f"  Mean accuracy: {metrics['mean_accuracy']:.4f}")
    if correct_key is not None:
        print(f"  Correct key accuracy: {metrics['correct_key_accuracy']:.4f}")
        print(f"  Key ranking: {metrics['key_ranking']}")
        print(f"  Success: {metrics['success']}")
    
    # Save results
    logger.save(f"CNN_MO_{dataset_name}_sh{max_shift}_training.json")
    
    # Plot results
    plot_accuracy_curves(history,
                        save_path=f"figures/desync_CNN_MO_{dataset_name}_accuracy.png",
                        title=f"CNN_MO on De-Synchronized {dataset_name}")
    
    if correct_key is not None:
        plot_branch_accuracies(metrics['branch_accuracies'], correct_key=correct_key,
                              save_path=f"figures/desync_CNN_MO_{dataset_name}_branches.png",
                              title=f"Branch Accuracies - CNN_MO on {dataset_name}")
    
    return history, metrics


if __name__ == "__main__":
    # Example usage - uncomment and modify paths as needed
    print("Example experiment scripts.")
    print("Please modify dataset paths and run specific experiments.")
    
    # Example:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # experiment_masking_mlp_mo("datasets/ASCAD", "Dataset2", shared_layer_size=200, device=device)





