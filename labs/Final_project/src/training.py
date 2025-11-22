"""
Training utilities for multi-output side-channel attack models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
import os

from mlp_mo import MLP_MO, MultiOutputLoss, compute_branch_accuracy
from cnn_mo import CNN_MO


class PowerTraceDataset(Dataset):
    """Dataset class for power traces and multi-output labels."""
    
    def __init__(self, traces: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            traces: Power traces of shape (n_traces, trace_length)
            labels: Multi-output labels of shape (n_traces, 256)
        """
        self.traces = torch.FloatTensor(traces)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str = "results/training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'branch_accuracies': [],  # Per-epoch accuracies for all 256 branches
            'attack_time': []
        }
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: Optional[float] = None, val_acc: Optional[float] = None,
                  branch_accuracies: Optional[np.ndarray] = None):
        """Log metrics for an epoch."""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss if val_loss is not None else -1)
        self.metrics['val_acc'].append(val_acc if val_acc is not None else -1)
        
        if branch_accuracies is not None:
            self.metrics['branch_accuracies'].append(branch_accuracies.tolist())
        else:
            self.metrics['branch_accuracies'].append([])
    
    def log_attack_time(self, attack_time: float):
        """Log total attack time."""
        self.metrics['attack_time'].append(attack_time)
    
    def save(self, filename: str):
        """Save metrics to JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filename: str):
        """Load metrics from JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)


def train_mlp_mo(model: MLP_MO, train_loader: DataLoader, val_loader: Optional[DataLoader],
                 num_epochs: int = 50, learning_rate: float = 0.001, 
                 device: str = 'cpu', logger: Optional[TrainingLogger] = None,
                 correct_key: Optional[int] = None) -> Dict:
    """
    Train MLP_MO model.
    
    Args:
        model: MLP_MO model instance
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        logger: Optional training logger
        correct_key: Optional correct key byte for monitoring
        
    Returns:
        Dictionary with training history and metrics
    """
    model = model.to(device)
    criterion = MultiOutputLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'correct_key_acc': [],
        'incorrect_key_acc': [],
        'branch_accuracies': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        for traces, labels in train_loader:
            traces = traces.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(traces)  # (batch_size, 256, 2)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute accuracies
            batch_accuracies = compute_branch_accuracy(outputs, labels)
            all_train_preds.append(outputs.detach().cpu())
            all_train_labels.append(labels.detach().cpu())
            
            train_total += labels.size(0)
        
        # Average training metrics
        avg_train_loss = train_loss / len(train_loader)
        
        # Compute overall accuracy (average across all branches)
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        all_accuracies = compute_branch_accuracy(all_train_preds, all_train_labels)
        avg_train_acc = all_accuracies.mean().item()
        
        # Validation phase
        val_loss = 0.0
        val_acc = 0.0
        all_val_preds = []
        all_val_labels = []
        
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for traces, labels in val_loader:
                    traces = traces.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(traces)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    all_val_preds.append(outputs.cpu())
                    all_val_labels.append(labels.cpu())
            
            avg_val_loss = val_loss / len(val_loader)
            all_val_preds = torch.cat(all_val_preds, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)
            val_accuracies = compute_branch_accuracy(all_val_preds, all_val_labels)
            avg_val_acc = val_accuracies.mean().item()
        else:
            avg_val_loss = None
            avg_val_acc = None
            val_accuracies = all_accuracies
        
        # Track correct key accuracy if available
        correct_key_acc = None
        incorrect_key_acc = None
        if correct_key is not None:
            correct_key_acc = val_accuracies[correct_key].item() if val_loader else all_accuracies[correct_key].item()
            # Average of incorrect keys
            incorrect_indices = [i for i in range(256) if i != correct_key]
            incorrect_key_acc = val_accuracies[incorrect_indices].mean().item() if val_loader else all_accuracies[incorrect_indices].mean().item()
        
        # Log metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['correct_key_acc'].append(correct_key_acc)
        history['incorrect_key_acc'].append(incorrect_key_acc)
        history['branch_accuracies'].append(val_accuracies.numpy().tolist() if val_loader else all_accuracies.numpy().tolist())
        
        if logger:
            logger.log_epoch(epoch, avg_train_loss, avg_train_acc, 
                           avg_val_loss, avg_val_acc, 
                           val_accuracies.numpy() if val_loader else all_accuracies.numpy())
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            if val_loader:
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            if correct_key is not None:
                print(f"  Correct Key Acc: {correct_key_acc:.4f}, Incorrect Key Acc: {incorrect_key_acc:.4f}")
    
    attack_time = time.time() - start_time
    history['attack_time'] = attack_time
    
    if logger:
        logger.log_attack_time(attack_time)
    
    print(f"\nTraining completed in {attack_time:.2f} seconds")
    
    return history


def train_cnn_mo(model: CNN_MO, train_loader: DataLoader, val_loader: Optional[DataLoader],
                num_epochs: int = 50, learning_rate: float = 0.001,
                device: str = 'cpu', logger: Optional[TrainingLogger] = None,
                correct_key: Optional[int] = None) -> Dict:
    """
    Train CNN_MO model.
    
    Args:
        model: CNN_MO model instance
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        logger: Optional training logger
        correct_key: Optional correct key byte for monitoring
        
    Returns:
        Dictionary with training history and metrics
    """
    # CNN_MO uses the same training procedure as MLP_MO
    from mlp_mo import MultiOutputLoss, compute_branch_accuracy
    
    model = model.to(device)
    criterion = MultiOutputLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'correct_key_acc': [],
        'incorrect_key_acc': [],
        'branch_accuracies': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for traces, labels in train_loader:
            traces = traces.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(traces)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            all_train_preds.append(outputs.detach().cpu())
            all_train_labels.append(labels.detach().cpu())
        
        avg_train_loss = train_loss / len(train_loader)
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        all_accuracies = compute_branch_accuracy(all_train_preds, all_train_labels)
        avg_train_acc = all_accuracies.mean().item()
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for traces, labels in val_loader:
                    traces = traces.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(traces)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    all_val_preds.append(outputs.cpu())
                    all_val_labels.append(labels.cpu())
            
            avg_val_loss = val_loss / len(val_loader)
            all_val_preds = torch.cat(all_val_preds, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)
            val_accuracies = compute_branch_accuracy(all_val_preds, all_val_labels)
            avg_val_acc = val_accuracies.mean().item()
        else:
            avg_val_loss = None
            avg_val_acc = None
            val_accuracies = all_accuracies
        
        # Track correct key accuracy
        correct_key_acc = None
        incorrect_key_acc = None
        if correct_key is not None:
            correct_key_acc = val_accuracies[correct_key].item() if val_loader else all_accuracies[correct_key].item()
            incorrect_indices = [i for i in range(256) if i != correct_key]
            incorrect_key_acc = val_accuracies[incorrect_indices].mean().item() if val_loader else all_accuracies[incorrect_indices].mean().item()
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['correct_key_acc'].append(correct_key_acc)
        history['incorrect_key_acc'].append(incorrect_key_acc)
        history['branch_accuracies'].append(val_accuracies.numpy().tolist() if val_loader else all_accuracies.numpy().tolist())
        
        if logger:
            logger.log_epoch(epoch, avg_train_loss, avg_train_acc,
                           avg_val_loss, avg_val_acc,
                           val_accuracies.numpy() if val_loader else all_accuracies.numpy())
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            if val_loader:
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            if correct_key is not None:
                print(f"  Correct Key Acc: {correct_key_acc:.4f}, Incorrect Key Acc: {incorrect_key_acc:.4f}")
    
    attack_time = time.time() - start_time
    history['attack_time'] = attack_time
    
    if logger:
        logger.log_attack_time(attack_time)
    
    print(f"\nTraining completed in {attack_time:.2f} seconds")
    
    return history

