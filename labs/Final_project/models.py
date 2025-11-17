"""
Multi-Output Neural Network Models for Side-Channel Attacks

This module implements MLPMO (Multi-Output MLP) and CNNMO (Multi-Output CNN)
models as described in the paper.
"""

import torch
import torch.nn as nn


class MLPMO(nn.Module):
    """
    Multi-Output Multi-Layer Perceptron for side-channel attacks.
    
    Architecture: Input → Shared Layer (optional) → 256 branches
    Each branch: 20×10-ReLU → 2-Softmax output
    
    Used for masking and noise-generation countermeasures.
    """
    
    def __init__(self, input_size, shared_layer_size=0, num_key_guesses=256):
        """
        Initialize MLPMO model.
        
        Args:
            input_size: Number of samples in power trace
            shared_layer_size: Size of shared layer (0 = no shared layer, Non-SoSL)
            num_key_guesses: Number of key hypotheses (default 256)
        """
        super(MLPMO, self).__init__()
        
        self.input_size = input_size
        self.shared_layer_size = shared_layer_size
        self.num_key_guesses = num_key_guesses
        
        # Shared layer (optional)
        if shared_layer_size > 0:
            self.shared_layer = nn.Sequential(
                nn.Linear(input_size, shared_layer_size),
                nn.ReLU()
            )
            branch_input_size = shared_layer_size
        else:
            self.shared_layer = None
            branch_input_size = input_size
        
        # Create 256 branches
        self.branches = nn.ModuleList()
        for _ in range(num_key_guesses):
            branch = nn.Sequential(
                nn.Linear(branch_input_size, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 2)  # Binary classification (LSB = 0 or 1)
            )
            self.branches.append(branch)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input power traces (batch_size, input_size)
            
        Returns:
            List of outputs for each branch, each with shape (batch_size, 2)
        """
        # Apply shared layer if present
        if self.shared_layer is not None:
            x = self.shared_layer(x)
        
        # Pass through each branch
        outputs = []
        for branch in self.branches:
            output = branch(x)
            outputs.append(output)
        
        return outputs


class CNNMO(nn.Module):
    """
    Multi-Output Convolutional Neural Network for side-channel attacks.
    
    Architecture: Input → Shared Conv Blocks → 256 branches
    Shared layers: 2 blocks of conv1d → batch norm → avg pool → ReLU
    
    Used for de-synchronization countermeasures.
    """
    
    def __init__(self, input_size, num_key_guesses=256, num_filters=64, kernel_size=3):
        """
        Initialize CNNMO model.
        
        Args:
            input_size: Number of samples in power trace
            num_key_guesses: Number of key hypotheses (default 256)
            num_filters: Number of filters in conv layers
            kernel_size: Size of convolution kernel
        """
        super(CNNMO, self).__init__()
        
        self.input_size = input_size
        self.num_key_guesses = num_key_guesses
        
        # Shared convolutional layers (2 blocks)
        # Block 1
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.relu2 = nn.ReLU()
        
        # Calculate size after conv blocks
        # After pool1: input_size // 2
        # After pool2: input_size // 4
        conv_output_size = (input_size // 4) * (num_filters * 2)
        
        # Create 256 branches
        self.branches = nn.ModuleList()
        for _ in range(num_key_guesses):
            branch = nn.Sequential(
                nn.Linear(conv_output_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # Binary classification
            )
            self.branches.append(branch)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input power traces (batch_size, input_size)
            
        Returns:
            List of outputs for each branch, each with shape (batch_size, 2)
        """
        # Reshape for conv1d: (batch_size, input_size) -> (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        # Flatten: (batch_size, num_filters*2, input_size//4) -> (batch_size, conv_output_size)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Pass through each branch
        outputs = []
        for branch in self.branches:
            output = branch(x)
            outputs.append(output)
        
        return outputs

