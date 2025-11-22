"""
Utility functions for vanilla GAN training and visualization.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(data_dir="./data", batch_size=128, num_workers=2):
    """
    Prepare MNIST dataset with appropriate transforms.
    
    Args:
        data_dir (str): Directory to store/download MNIST data
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for MNIST training data
    """
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # Transform: Convert to tensor and normalize from [0,1] to [-1,1] for Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Download and load MNIST training data
    train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers
    )
    
    return train_loader


def generate_noise(batch_size, noise_dim, device):
    """
    Generate random noise vector for generator input.
    
    Args:
        batch_size (int): Number of noise vectors to generate
        noise_dim (int): Dimension of each noise vector
        device (torch.device): Device to create tensor on
        
    Returns:
        torch.Tensor: Random noise tensor of shape (batch_size, noise_dim)
    """
    return torch.randn(batch_size, noise_dim, device=device)


def denormalize_images(images):
    """
    Denormalize images from [-1, 1] back to [0, 1] for display.
    
    Args:
        images (torch.Tensor): Normalized images in range [-1, 1]
        
    Returns:
        torch.Tensor: Denormalized images in range [0, 1]
    """
    return (images + 1) / 2.0


def save_generated_images(generator, num_images=64, noise_dim=100, device=None, 
                         save_path=None, nrow=8):
    """
    Generate and save a grid of images from the generator.
    
    Args:
        generator (nn.Module): Trained generator model
        num_images (int): Number of images to generate
        noise_dim (int): Dimension of noise vector
        device (torch.device): Device to run generation on
        save_path (str): Path to save the image (if None, just display)
        nrow (int): Number of images per row in the grid
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate noise
        noise = generate_noise(num_images, noise_dim, device)
        
        # Generate images
        fake_images = generator(noise)
        
        # Move to CPU and denormalize
        fake_images = fake_images.cpu()
        fake_images = denormalize_images(fake_images)
        fake_images = torch.clamp(fake_images, 0, 1)
        
        # Create grid
        grid = torchvision.utils.make_grid(fake_images, nrow=nrow, pad_value=1.0)
        
        # Convert to numpy for plotting
        npimg = grid.numpy()
        if len(npimg.shape) == 3:
            npimg = npimg.transpose(1, 2, 0)
        
        # Plot
        plt.figure(figsize=(12, 12))
        if len(npimg.shape) == 3 and npimg.shape[2] == 1:
            plt.imshow(npimg.squeeze(2), cmap='gray')
        else:
            plt.imshow(npimg)
        plt.axis('off')
        plt.title(f'Generated MNIST Images (n={num_images})', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved generated images to {save_path}")
        else:
            plt.show()
    
    generator.train()


def plot_training_losses(disc_losses, gen_losses, save_path=None):
    """
    Plot discriminator and generator training losses over time.
    
    Args:
        disc_losses (list): List of discriminator losses
        gen_losses (list): List of generator losses
        save_path (str): Path to save the plot (if None, just display)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(disc_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(gen_losses, label='Generator Loss', alpha=0.7)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved training losses plot to {save_path}")
    else:
        plt.show()


def get_device():
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device: Device for training
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


