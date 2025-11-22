"""
Convolutional Neural Network Multi-Output (CNN_MO) architecture.

Designed for de-synchronization countermeasures using translation-invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNN_MO(nn.Module):
    """
    Multi-Output CNN for side-channel attacks with de-synchronization.
    
    Architecture:
    - Input Layer: (trace_length, 1)
    - Shared Layers:
      - Block 1: Conv1D → BatchNorm → AvgPool → ReLU
      - Block 2: Conv1D → BatchNorm → AvgPool → ReLU
    - 256 Branches (parallel):
      - Each branch processes flattened features
    """
    
    def __init__(self, trace_length: int, num_filters1: int = 32, 
                 num_filters2: int = 64, kernel_size: int = 3,
                 pool_size: int = 2, hidden_size: int = 128):
        """
        Initialize CNN_MO model.
        
        Args:
            trace_length: Length of input power traces
            num_filters1: Number of filters in first conv layer (default: 32)
            num_filters2: Number of filters in second conv layer (default: 64)
            kernel_size: Size of convolution kernel (default: 3)
            pool_size: Size of average pooling (default: 2)
            hidden_size: Size of hidden layer in branches (default: 128)
        """
        super(CNN_MO, self).__init__()
        
        self.trace_length = trace_length
        self.num_branches = 256
        
        # Shared convolutional layers
        # Input: (batch, 1, trace_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters1, 
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters1)
        self.pool1 = nn.AvgPool1d(kernel_size=pool_size)
        
        # Compute size after first conv block
        conv1_out_length = trace_length // pool_size
        
        self.conv2 = nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2,
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters2)
        self.pool2 = nn.AvgPool1d(kernel_size=pool_size)
        
        # Compute size after second conv block
        conv2_out_length = conv1_out_length // pool_size
        flattened_size = num_filters2 * conv2_out_length
        
        # Create 256 parallel branches
        self.branches = nn.ModuleList()
        for _ in range(self.num_branches):
            branch = nn.Sequential(
                nn.Linear(flattened_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2),  # Binary classification (LSB = 0 or 1)
                nn.Softmax(dim=-1)
            )
            self.branches.append(branch)
        
        self.flattened_size = flattened_size
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        # Initialize conv layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # Initialize branches
        for branch in self.branches:
            for module in branch:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input traces of shape (batch_size, trace_length)
               Will be reshaped to (batch_size, 1, trace_length) for conv layers
            
        Returns:
            Output tensor of shape (batch_size, 256, 2)
        """
        batch_size = x.size(0)
        
        # Reshape for conv1d: (batch, 1, trace_length)
        x = x.unsqueeze(1)
        
        # Shared convolutional layers
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten: (batch, num_filters2, conv2_out_length) -> (batch, flattened_size)
        x = x.view(batch_size, -1)
        
        # Process through all 256 branches
        outputs = []
        for branch in self.branches:
            branch_output = branch(x)  # (batch_size, 2)
            outputs.append(branch_output)
        
        # Stack outputs: (batch_size, 256, 2)
        return torch.stack(outputs, dim=1)
    
    def predict_key(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the most likely key based on branch outputs.
        
        Args:
            x: Input traces of shape (batch_size, trace_length)
            
        Returns:
            Predicted key indices of shape (batch_size,)
        """
        outputs = self.forward(x)  # (batch_size, 256, 2)
        
        # Get probabilities for class 1 (LSB=1)
        prob_lsb1 = outputs[:, :, 1]  # (batch_size, 256)
        
        # The key with highest probability is our prediction
        predicted_keys = torch.argmax(prob_lsb1, dim=1)
        
        return predicted_keys





