"""
Evaluation metrics for side-channel attack experiments.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from mlp_mo import compute_branch_accuracy


def compute_success_rate(predicted_keys: List[int], correct_key: int) -> float:
    """
    Compute success rate of attacks.
    
    Args:
        predicted_keys: List of predicted keys from multiple attacks
        correct_key: Correct key byte
        
    Returns:
        Success rate (0.0 to 1.0)
    """
    if len(predicted_keys) == 0:
        return 0.0
    
    correct_predictions = sum(1 for key in predicted_keys if key == correct_key)
    return correct_predictions / len(predicted_keys)


def compute_key_ranking(branch_accuracies: np.ndarray, correct_key: int) -> int:
    """
    Compute the ranking of the correct key based on branch accuracies.
    
    Args:
        branch_accuracies: Accuracies for all 256 key hypotheses
        correct_key: Correct key byte
        
    Returns:
        Ranking (1 = best, 256 = worst)
    """
    # Ensure correct_key is a plain Python int
    correct_key = int(correct_key)
    
    # Sort keys by accuracy (descending)
    sorted_indices = np.argsort(branch_accuracies)[::-1]
    
    # Ensure sorted_indices is a regular numeric array before comparison
    sorted_indices = np.asarray(sorted_indices, dtype=np.int64)
    
    # Find position of correct key
    ranking = np.where(sorted_indices == correct_key)[0][0] + 1
    return ranking


def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  device: str = 'cpu', correct_key: Optional[int] = None) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        correct_key: Optional correct key for detailed metrics
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in data_loader:
            traces = traces.to(device)
            labels = labels.to(device)
            
            outputs = model(traces)  # (batch_size, 256, 2)
            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute branch accuracies
    branch_accuracies = compute_branch_accuracy(all_predictions, all_labels)
    
    # Get predicted keys (branch with highest accuracy)
    predicted_keys = torch.argmax(branch_accuracies).item()
    
    metrics = {
        'branch_accuracies': branch_accuracies.numpy(),
        'mean_accuracy': branch_accuracies.mean().item(),
        'predicted_key': predicted_keys
    }
    
    if correct_key is not None:
        # Ensure correct_key is a plain Python int to avoid dtype comparison issues
        correct_key = int(correct_key)
        metrics['correct_key_accuracy'] = branch_accuracies[correct_key].item()
        metrics['key_ranking'] = compute_key_ranking(branch_accuracies.numpy(), correct_key)
        metrics['success'] = (predicted_keys == correct_key)
        
        # Compute gap between correct and incorrect keys
        incorrect_indices = [i for i in range(256) if i != correct_key]
        incorrect_key_acc = branch_accuracies[incorrect_indices].mean().item()
        metrics['accuracy_gap'] = metrics['correct_key_accuracy'] - incorrect_key_acc
    
    return metrics


def run_repeated_attacks(model_class, model_kwargs: Dict, train_loader, val_loader,
                        num_attacks: int = 50, num_epochs: int = 50,
                        correct_key: Optional[int] = None, device: str = 'cpu') -> Dict:
    """
    Run multiple attacks with different random seeds to compute success rate.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_attacks: Number of repeated attacks
        num_epochs: Number of epochs per attack
        correct_key: Correct key byte
        device: Device to train on
        
    Returns:
        Dictionary with aggregated results
    """
    from .training import train_mlp_mo, train_cnn_mo
    
    predicted_keys = []
    attack_times = []
    all_branch_accuracies = []
    
    # Determine training function based on model class
    if 'MLP' in model_class.__name__:
        train_func = train_mlp_mo
    elif 'CNN' in model_class.__name__:
        train_func = train_cnn_mo
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    for attack_idx in range(num_attacks):
        print(f"\nRunning attack {attack_idx + 1}/{num_attacks}")
        
        # Set random seed for reproducibility
        torch.manual_seed(attack_idx)
        np.random.seed(attack_idx)
        
        # Create new model instance
        model = model_class(**model_kwargs)
        
        # Train model
        history = train_func(model, train_loader, val_loader, num_epochs=num_epochs,
                           device=device, correct_key=correct_key)
        
        # Evaluate on validation set (avoid circular import by calling directly)
        metrics = evaluate_model(model, val_loader, device=device, correct_key=correct_key)
        
        predicted_keys.append(metrics['predicted_key'])
        attack_times.append(history['attack_time'])
        all_branch_accuracies.append(metrics['branch_accuracies'])
    
    # Aggregate results
    results = {
        'predicted_keys': predicted_keys,
        'attack_times': attack_times,
        'mean_attack_time': np.mean(attack_times),
        'std_attack_time': np.std(attack_times),
        'all_branch_accuracies': all_branch_accuracies
    }
    
    if correct_key is not None:
        results['success_rate'] = compute_success_rate(predicted_keys, correct_key)
        results['correct_key'] = correct_key
    
    return results


def compare_with_baseline(mo_results: Dict, baseline_results: Dict) -> Dict:
    """
    Compare multi-output results with baseline (e.g., DDLA).
    
    Args:
        mo_results: Results from multi-output model
        baseline_results: Results from baseline model
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'speedup': baseline_results.get('attack_time', 1) / mo_results.get('mean_attack_time', 1),
        'success_rate_improvement': mo_results.get('success_rate', 0) - baseline_results.get('success_rate', 0),
        'mo_attack_time': mo_results.get('mean_attack_time'),
        'baseline_attack_time': baseline_results.get('attack_time')
    }
    
    return comparison

