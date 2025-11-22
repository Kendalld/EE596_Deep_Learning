"""
Vanilla GAN Models for MNIST
Generator and Discriminator architectures using fully connected layers.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Vanilla GAN Generator for MNIST.
    
    Takes random noise as input and generates 28x28 grayscale images.
    Architecture: Fully connected layers with ReLU activations and Tanh output.
    
    Args:
        input_noise_dim (int): Dimension of the input noise vector (default: 100)
        image_size (int): Size of the output image in pixels (28x28 = 784 for MNIST)
        hidden_dims (list): List of hidden layer dimensions (default: [256, 512, 1024])
    """
    
    def __init__(self, input_noise_dim=100, image_size=784, hidden_dims=[256, 512, 1024]):
        super(Generator, self).__init__()
        
        self.input_noise_dim = input_noise_dim
        self.image_size = image_size
        
        # Build fully connected layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_noise_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer with Tanh activation (outputs in range [-1, 1])
        layers.append(nn.Linear(prev_dim, image_size))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Input noise tensor of shape (batch_size, input_noise_dim)
            
        Returns:
            torch.Tensor: Generated images of shape (batch_size, 1, 28, 28)
        """
        out = self.model(x)
        # Reshape to image dimensions (batch_size, 1, 28, 28) for MNIST
        out = out.view(-1, 1, 28, 28)
        return out


class Discriminator(nn.Module):
    """
    Vanilla GAN Discriminator for MNIST.
    
    Takes 28x28 grayscale images as input and outputs a probability of being real.
    Architecture: Fully connected layers with LeakyReLU activations and Sigmoid output.
    
    Args:
        image_size (int): Size of the input image in pixels (28x28 = 784 for MNIST)
        hidden_dims (list): List of hidden layer dimensions (default: [1024, 512, 256])
        dropout_prob (float): Dropout probability for regularization (default: 0.3)
        leaky_relu_slope (float): Negative slope for LeakyReLU (default: 0.2)
    """
    
    def __init__(self, image_size=784, hidden_dims=[1024, 512, 256], 
                 dropout_prob=0.3, leaky_relu_slope=0.2):
        super(Discriminator, self).__init__()
        
        self.image_size = image_size
        
        # Build fully connected layers dynamically based on hidden_dims
        layers = []
        prev_dim = image_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim
        
        # Output layer with Sigmoid activation (outputs probability in range [0, 1])
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 1, 28, 28) or (batch_size, 784)
            
        Returns:
            torch.Tensor: Probability scores of shape (batch_size,)
        """
        # Flatten the input image if needed
        if x.dim() > 2:
            x = x.view(-1, self.image_size)
        
        out = self.model(x)
        # Remove unnecessary dimension to get (batch_size,)
        return out.squeeze(1)


