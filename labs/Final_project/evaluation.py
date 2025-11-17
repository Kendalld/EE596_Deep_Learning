"""
Evaluation Metrics, Plotting, and Attack Success Calculation

This module provides utilities for evaluating side-channel attack performance,
plotting results, and calculating success rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import time


def calculate_success_rate(predicted_keys, correct_key, num_runs):
    """
    Calculate success rate of attacks.
    
    Args:
        predicted_keys: List of predicted keys from multiple runs
        correct_key: Correct key byte value
        num_runs: Number of attack runs
        
    Returns:
        Success rate (percentage)
    """
    if len(predicted_keys) != num_runs:
        raise ValueError(f"Number of predicted keys ({len(predicted_keys)}) doesn't match num_runs ({num_runs})")
    
    successful = sum(1 for key in predicted_keys if key == correct_key)
    success_rate = (successful / num_runs) * 100.0
    
    return success_rate


def plot_accuracy_curves(history, correct_key, save_path=None, title="Accuracy Curves"):
    """
    Plot accuracy curves showing correct key (red) vs incorrect keys.
    
    Args:
        history: Training history dictionary with 'correct_key_accuracies' and 'incorrect_key_accuracies'
        correct_key: Correct key byte value
        save_path: Path to save figure (optional)
        title: Plot title
    """
    epochs = range(1, len(history['correct_key_accuracies']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['correct_key_accuracies'], 'r-', linewidth=2, 
             label=f'Correct Key (Key={correct_key})')
    plt.plot(epochs, history['incorrect_key_accuracies'], 'b--', linewidth=1, 
             alpha=0.7, label='Incorrect Keys (Average)')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_all_key_accuracies(accuracies, correct_key, epoch=None, save_path=None):
    """
    Plot accuracy for all 256 key hypotheses.
    
    Args:
        accuracies: Array of accuracies for each key hypothesis (256,)
        correct_key: Correct key byte value
        epoch: Epoch number (for title)
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 6))
    
    key_indices = np.arange(256)
    
    # Plot all keys
    plt.plot(key_indices, accuracies, 'b-', linewidth=0.5, alpha=0.5, label='All Keys')
    
    # Highlight correct key
    plt.plot(correct_key, accuracies[correct_key], 'ro', markersize=10, 
             label=f'Correct Key (Key={correct_key})')
    
    plt.xlabel('Key Hypothesis', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    title = f'Accuracy per Key Hypothesis'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 255)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attack_time_comparison(results_dict, save_path=None):
    """
    Plot attack time comparison between different models.
    
    Args:
        results_dict: Dictionary with model names as keys and attack times as values
        save_path: Path to save figure (optional)
    """
    models = list(results_dict.keys())
    times = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Attack Time (seconds)', fontsize=12)
    plt.title('Attack Time Comparison', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_success_rate_comparison(results_dict, noise_levels, save_path=None):
    """
    Plot success rate comparison for different noise levels.
    
    Args:
        results_dict: Dictionary with model names as keys and success rates per noise level as values
        noise_levels: List of noise levels (sigma values)
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, success_rates in results_dict.items():
        plt.plot(noise_levels, success_rates, marker='o', linewidth=2, label=model_name)
    
    plt.xlabel('Noise Level (σ)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Success Rate vs Noise Level', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_comparison_table(results_dict, metric_name="Metric"):
    """
    Create a comparison table for different models.
    
    Args:
        results_dict: Dictionary with model names as keys and metric values as values
        metric_name: Name of the metric
        
    Returns:
        Formatted table string
    """
    table = f"\n{metric_name} Comparison:\n"
    table += "-" * 50 + "\n"
    table += f"{'Model':<30} {metric_name:>15}\n"
    table += "-" * 50 + "\n"
    
    for model_name, value in results_dict.items():
        if isinstance(value, float):
            table += f"{model_name:<30} {value:>15.4f}\n"
        else:
            table += f"{model_name:<30} {value:>15}\n"
    
    table += "-" * 50 + "\n"
    
    return table


def measure_attack_time(model, dataloader, device, num_epochs=10):
    """
    Measure attack time for a model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for training
        device: Device
        num_epochs: Number of epochs to train
        
    Returns:
        Total attack time in seconds
    """
    from training import train_model
    
    start_time = time.time()
    
    # Create a dummy validation loader (can be same as training for timing)
    history = train_model(model, dataloader, dataloader, num_epochs, device, verbose=False)
    
    total_time = time.time() - start_time
    
    return total_time


def calculate_speedup(baseline_time, improved_time):
    """
    Calculate speedup factor.
    
    Args:
        baseline_time: Baseline attack time
        improved_time: Improved attack time
        
    Returns:
        Speedup factor
    """
    return baseline_time / improved_time

