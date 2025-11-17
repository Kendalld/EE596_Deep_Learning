"""
Training Loops with Multi-Loss Computation

This module implements training loops for multi-output models with
multi-loss computation: L_total = Σ(γ_k * L[k](θ)) for k=1 to 256
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# tqdm is optional - only used if available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a dummy tqdm function if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


def compute_multi_loss(outputs, labels, loss_fn, gamma_weights=None):
    """
    Compute multi-loss for all branches.
    
    Formula: L_total = Σ(γ_k * L[k](θ)) for k=1 to 256
    
    Args:
        outputs: List of outputs from each branch, each shape (batch_size, 2)
        labels: Ground truth labels (batch_size, num_key_guesses)
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        gamma_weights: Weight factors for each branch (default: all 1.0)
        
    Returns:
        Total loss (scalar tensor)
    """
    num_branches = len(outputs)
    batch_size = outputs[0].size(0)
    
    if gamma_weights is None:
        gamma_weights = torch.ones(num_branches, device=outputs[0].device)
    
    total_loss = 0.0
    
    for k in range(num_branches):
        # Get output for branch k: (batch_size, 2)
        branch_output = outputs[k]
        
        # Get labels for branch k: (batch_size,)
        branch_labels = labels[:, k].long()
        
        # Compute loss for branch k
        branch_loss = loss_fn(branch_output, branch_labels)
        
        # Weighted sum
        total_loss += gamma_weights[k] * branch_loss
    
    return total_loss


def train_epoch(model, dataloader, loss_fn, optimizer, device, gamma_weights=None):
    """
    Train model for one epoch.
    
    Args:
        model: Multi-output model (MLPMO or CNNMO)
        dataloader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device (cpu or cuda)
        gamma_weights: Weight factors for each branch
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for traces, labels in dataloader:
        traces = traces.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(traces)
        
        # Compute multi-loss
        loss = compute_multi_loss(outputs, labels, loss_fn, gamma_weights)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_accuracy(model, dataloader, device, correct_key=None):
    """
    Evaluate model accuracy per key hypothesis.
    
    Args:
        model: Multi-output model
        dataloader: DataLoader for evaluation data
        device: Device
        correct_key: Correct key byte value (0-255) if known, for highlighting
        
    Returns:
        Dictionary with accuracy per key hypothesis and overall metrics
    """
    model.eval()
    num_key_guesses = len(model.branches)
    correct_predictions = torch.zeros(num_key_guesses, device=device)
    total_samples = 0
    
    with torch.no_grad():
        for traces, labels in dataloader:
            traces = traces.to(device)
            labels = labels.to(device)
            batch_size = traces.size(0)
            
            # Forward pass
            outputs = model(traces)
            
            # Compute accuracy for each branch
            for k in range(num_key_guesses):
                branch_output = outputs[k]
                branch_labels = labels[:, k]
                
                # Get predictions
                predictions = torch.argmax(branch_output, dim=1)
                
                # Count correct predictions
                correct_predictions[k] += (predictions == branch_labels).sum().item()
            
            total_samples += batch_size
    
    # Calculate accuracy per key hypothesis
    accuracies = (correct_predictions / total_samples).cpu().numpy()
    
    results = {
        'accuracies': accuracies,
        'total_samples': total_samples
    }
    
    if correct_key is not None:
        results['correct_key_accuracy'] = accuracies[correct_key]
        results['incorrect_key_accuracy'] = np.mean(accuracies[np.arange(num_key_guesses) != correct_key])
        results['accuracy_gap'] = results['correct_key_accuracy'] - results['incorrect_key_accuracy']
    
    return results


def train_model(model, train_loader, val_loader, num_epochs, device, 
                learning_rate=0.001, correct_key=None, verbose=True):
    """
    Train multi-output model with full training loop.
    
    Args:
        model: Multi-output model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_epochs: Number of training epochs
        device: Device
        learning_rate: Learning rate for Adam optimizer
        correct_key: Correct key byte value if known
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_accuracies': [],
        'correct_key_accuracies': [],
        'incorrect_key_accuracies': [],
        'accuracy_gaps': [],
        'epoch_times': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Evaluate
        val_results = evaluate_accuracy(model, val_loader, device, correct_key)
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_accuracies'].append(val_results['accuracies'])
        
        if correct_key is not None:
            history['correct_key_accuracies'].append(val_results['correct_key_accuracy'])
            history['incorrect_key_accuracies'].append(val_results['incorrect_key_accuracy'])
            history['accuracy_gaps'].append(val_results['accuracy_gap'])
        
        history['epoch_times'].append(epoch_time)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if correct_key is not None:
                print(f"  Correct Key Accuracy: {val_results['correct_key_accuracy']:.4f}")
                print(f"  Incorrect Key Accuracy: {val_results['incorrect_key_accuracy']:.4f}")
                print(f"  Accuracy Gap: {val_results['accuracy_gap']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print()
    
    total_time = time.time() - start_time
    history['total_time'] = total_time
    
    return history


def identify_correct_key(model, dataloader, device):
    """
    Identify the correct key by finding the branch with highest accuracy.
    
    Args:
        model: Trained multi-output model
        dataloader: DataLoader for evaluation
        device: Device
        
    Returns:
        Predicted key byte value (0-255)
    """
    results = evaluate_accuracy(model, dataloader, device)
    accuracies = results['accuracies']
    
    # Key with highest accuracy is the predicted correct key
    predicted_key = np.argmax(accuracies)
    
    return predicted_key, accuracies

