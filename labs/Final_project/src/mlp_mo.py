"""
Multi-Layer Perceptron Multi-Output (MLP_MO) architecture.

Implements both Non-SoSL (no shared layer) and SoSL (shared layer) variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLP_MO(nn.Module):
    """
    Multi-Output MLP for side-channel attacks.
    
    Architecture:
    - Input Layer: (trace_length,)
    - Optional Shared Layer: shared_layer_size nodes
    - 256 Branches (parallel):
      - Hidden Layer 1: 20 nodes, ReLU
      - Hidden Layer 2: 10 nodes, ReLU
      - Output Layer: 2 nodes, Softmax
    """
    
    def __init__(self, trace_length: int, shared_layer_size: int = 0, 
                 hidden1_size: int = 20, hidden2_size: int = 10):
        """
        Initialize MLP_MO model.
        
        Args:
            trace_length: Length of input power traces
            shared_layer_size: Size of shared layer (0 for Non-SoSL, 200 for SoSL-200)
            hidden1_size: Size of first hidden layer in each branch (default: 20)
            hidden2_size: Size of second hidden layer in each branch (default: 10)
        """
        super(MLP_MO, self).__init__()
        
        self.trace_length = trace_length
        self.shared_layer_size = shared_layer_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.num_branches = 256  # One branch per key hypothesis
        
        # Shared layer (optional)
        if shared_layer_size > 0:
            self.shared_layer = nn.Sequential(
                nn.Linear(trace_length, shared_layer_size),
                nn.ReLU()
            )
            branch_input_size = shared_layer_size
        else:
            self.shared_layer = None
            branch_input_size = trace_length
        
        # Create 256 parallel branches
        self.branches = nn.ModuleList()
        for _ in range(self.num_branches):
            branch = nn.Sequential(
                nn.Linear(branch_input_size, hidden1_size),
                nn.ReLU(),
                nn.Linear(hidden1_size, hidden2_size),
                nn.ReLU(),
                nn.Linear(hidden2_size, 2),  # Binary classification (LSB = 0 or 1)
                nn.Softmax(dim=-1)
            )
            self.branches.append(branch)
        
        # Initialize weights equivalently for all branches
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights equivalently for all branches."""
        for branch in self.branches:
            for module in branch:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        if self.shared_layer is not None:
            for module in self.shared_layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input traces of shape (batch_size, trace_length)
            
        Returns:
            Output tensor of shape (batch_size, 256, 2)
            Each branch outputs 2 probabilities (for LSB=0 and LSB=1)
        """
        batch_size = x.size(0)
        
        # Apply shared layer if present
        if self.shared_layer is not None:
            x = self.shared_layer(x)
        
        # Process through all 256 branches
        outputs = []
        for branch in self.branches:
            branch_output = branch(x)  # (batch_size, 2)
            outputs.append(branch_output)
        
        # Stack outputs: (batch_size, 256, 2)
        return torch.stack(outputs, dim=1)
    
    def predict_key(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the most likely key based on branch accuracies.
        
        Args:
            x: Input traces of shape (batch_size, trace_length)
            
        Returns:
            Predicted key indices of shape (batch_size,)
        """
        outputs = self.forward(x)  # (batch_size, 256, 2)
        
        # For each branch, get the probability of the predicted class
        # We use the maximum probability as the confidence
        # The correct key should have higher accuracy (probability of correct LSB)
        # For now, we'll use the branch with highest average confidence
        # In practice, we compare with ground truth labels during training
        
        # Get probabilities for class 1 (LSB=1)
        prob_lsb1 = outputs[:, :, 1]  # (batch_size, 256)
        
        # The key with highest probability is our prediction
        # But this is simplified - actual key recovery uses accuracy on validation set
        predicted_keys = torch.argmax(prob_lsb1, dim=1)
        
        return predicted_keys


class MultiOutputLoss(nn.Module):
    """
    Multi-loss function for multi-output classification.
    
    Each of 256 branches has its own loss:
    L[k](θ) = -1/Ns * Σ(y_true * ln(z))
    
    Total loss: L_total = Σ(γ_k * L[k](θ)) where γ_k = 1
    """
    
    def __init__(self):
        super(MultiOutputLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction='mean')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-output loss.
        
        Args:
            predictions: Model outputs of shape (batch_size, 256, 2)
                        Each branch outputs softmax probabilities
            targets: Ground truth labels of shape (batch_size, 256)
                     Each column is the LSB label for that key hypothesis
                     
        Returns:
            Total loss (scalar)
        """
        batch_size, num_branches, num_classes = predictions.shape
        
        # Convert predictions to log probabilities for NLLLoss
        log_probs = torch.log(predictions + 1e-8)  # Add small epsilon to avoid log(0)
        
        # Compute loss for each branch
        total_loss = 0.0
        for k in range(num_branches):
            # Get predictions and targets for branch k
            branch_log_probs = log_probs[:, k, :]  # (batch_size, 2)
            branch_targets = targets[:, k].long()  # (batch_size,)
            
            # Compute cross-entropy loss for this branch
            branch_loss = F.nll_loss(branch_log_probs, branch_targets, reduction='mean')
            
            # Add to total loss (γ_k = 1 for all branches)
            total_loss += branch_loss
        
        return total_loss / num_branches  # Average across branches


def compute_branch_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy for each branch (key hypothesis).
    
    Args:
        predictions: Model outputs of shape (batch_size, 256, 2)
        targets: Ground truth labels of shape (batch_size, 256)
        
    Returns:
        Accuracies of shape (256,) - one accuracy per key hypothesis
    """
    batch_size, num_branches, num_classes = predictions.shape
    
    # Get predicted classes (argmax of softmax probabilities)
    predicted_classes = torch.argmax(predictions, dim=2)  # (batch_size, 256)
    
    # Compute accuracy for each branch
    accuracies = []
    for k in range(num_branches):
        correct = (predicted_classes[:, k] == targets[:, k]).float()
        accuracy = correct.mean()
        accuracies.append(accuracy.item())
    
    return torch.tensor(accuracies)

