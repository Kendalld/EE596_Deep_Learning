"""
Visualization utilities for side-channel attack results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import os


def plot_accuracy_curves(history: Dict, save_path: Optional[str] = None, 
                        title: str = "Training Accuracy Curves"):
    """
    Plot accuracy curves for correct vs incorrect keys.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
        title: Plot title
    """
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(10, 6))
    
    if history.get('correct_key_acc') and history['correct_key_acc'][0] is not None:
        plt.plot(epochs, history['correct_key_acc'], label='Correct Key', linewidth=2)
        plt.plot(epochs, history['incorrect_key_acc'], label='Incorrect Keys (avg)', linewidth=2)
    
    plt.plot(epochs, history['train_acc'], label='Train Accuracy (avg)', linestyle='--', alpha=0.7)
    
    if history.get('val_acc') and history['val_acc'][0] is not None:
        plt.plot(epochs, history['val_acc'], label='Val Accuracy (avg)', linestyle='--', alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_attack_time_comparison(results: Dict, save_path: Optional[str] = None,
                               title: str = "Attack Time Comparison"):
    """
    Plot attack time comparison between models.
    
    Args:
        results: Dictionary with attack times for different models
        save_path: Optional path to save figure
        title: Plot title
    """
    models = list(results.keys())
    times = [results[model] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.ylabel('Attack Time (seconds)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_success_rate_comparison(results: Dict, save_path: Optional[str] = None,
                                title: str = "Success Rate Comparison"):
    """
    Plot success rate comparison for different noise levels or models.
    
    Args:
        results: Dictionary with success rates (keys are model names or noise levels)
        save_path: Optional path to save figure
        title: Plot title
    """
    models = list(results.keys())
    success_rates = [results[model] * 100 for model in models]  # Convert to percentage
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_branch_accuracies(branch_accuracies: np.ndarray, correct_key: Optional[int] = None,
                          save_path: Optional[str] = None, title: str = "Branch Accuracies"):
    """
    Plot accuracies for all 256 key hypotheses.
    
    Args:
        branch_accuracies: Array of accuracies for 256 branches
        correct_key: Optional correct key to highlight
        save_path: Optional path to save figure
        title: Plot title
    """
    plt.figure(figsize=(14, 6))
    
    keys = np.arange(256)
    plt.plot(keys, branch_accuracies, 'b-', linewidth=1, alpha=0.7, label='All Keys')
    
    if correct_key is not None:
        plt.axvline(x=correct_key, color='r', linestyle='--', linewidth=2, label=f'Correct Key ({correct_key})')
        plt.scatter([correct_key], [branch_accuracies[correct_key]], 
                   color='r', s=100, zorder=5, label='Correct Key')
    
    plt.xlabel('Key Hypothesis', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_noise_comparison(noise_results: Dict, save_path: Optional[str] = None,
                         title: str = "Noise Robustness Comparison"):
    """
    Plot success rate vs noise level for different models.
    
    Args:
        noise_results: Dictionary with structure {model_name: {sigma: success_rate}}
        save_path: Optional path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    sigmas = sorted(set(sigma for model_results in noise_results.values() 
                       for sigma in model_results.keys()))
    
    for model_name, model_results in noise_results.items():
        success_rates = [model_results.get(sigma, 0) * 100 for sigma in sigmas]
        plt.plot(sigmas, success_rates, marker='o', linewidth=2, label=model_name)
    
    plt.xlabel('Noise Level (σ)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_desync_comparison(results: Dict, save_path: Optional[str] = None,
                          title: str = "De-Synchronization Attack Comparison"):
    """
    Plot comparison of different attack methods on de-synchronized data.
    
    Args:
        results: Dictionary with attack results (e.g., {'CPA': ..., 'CNN_DDLA': ..., 'CNN_MO': ...})
        save_path: Optional path to save figure
        title: Plot title
    """
    methods = list(results.keys())
    
    # Extract metrics (assuming results contain 'attack_time' and 'success_rate')
    attack_times = [results[method].get('attack_time', 0) for method in methods]
    success_rates = [results[method].get('success_rate', 0) * 100 for method in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Attack time comparison
    bars1 = ax1.bar(methods, attack_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Attack Time (seconds)', fontsize=12)
    ax1.set_title('Attack Time', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars1, attack_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # Success rate comparison
    bars2 = ax2.bar(methods, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars2, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()





