"""
Vanilla GAN Training Script for MNIST
Main training loop for training a vanilla GAN on the MNIST dataset.
"""

import torch
import torch.nn as nn
import argparse
import os
from models import Generator, Discriminator
from utils import (
    prepare_data, generate_noise, get_device, set_seed,
    save_generated_images, plot_training_losses
)


def train_gan(
    epochs=50,
    batch_size=128,
    noise_dim=100,
    lr_generator=0.0002,
    lr_discriminator=0.0001,
    label_smooth_real=0.9,
    label_smooth_fake=0.1,
    gen_train_steps=1,
    data_dir="./data",
    save_dir="./checkpoints",
    seed=42,
    device=None
):
    """
    Train a vanilla GAN on MNIST dataset.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        noise_dim (int): Dimension of noise vector for generator
        lr_generator (float): Learning rate for generator optimizer
        lr_discriminator (float): Learning rate for discriminator optimizer
        label_smooth_real (float): Label smoothing value for real images (default: 0.9)
        label_smooth_fake (float): Label smoothing value for fake images (default: 0.1)
        gen_train_steps (int): Number of generator training steps per discriminator step
        data_dir (str): Directory for MNIST data
        save_dir (str): Directory to save checkpoints and outputs
        seed (int): Random seed for reproducibility
        device (torch.device): Device to train on (auto-detected if None)
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Get device
    if device is None:
        device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Prepare data
    print("Loading MNIST dataset...")
    train_loader = prepare_data(data_dir=data_dir, batch_size=batch_size)
    print(f"Dataset loaded. Number of batches per epoch: {len(train_loader)}")
    
    # Initialize models
    print("Initializing Generator and Discriminator...")
    generator = Generator(input_noise_dim=noise_dim, image_size=28*28)
    discriminator = Discriminator(image_size=28*28)
    
    generator.to(device)
    discriminator.to(device)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizer_gen = torch.optim.Adam(
        generator.parameters(), 
        lr=lr_generator, 
        betas=(0.5, 0.999)
    )
    optimizer_disc = torch.optim.Adam(
        discriminator.parameters(), 
        lr=lr_discriminator, 
        betas=(0.5, 0.999)
    )
    
    # Track losses
    disc_losses = []
    gen_losses = []
    
    print("\nStarting training...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(epochs):
        running_loss_d = 0.0
        running_loss_g = 0.0
        num_batches = 0
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            current_batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Create labels with label smoothing
            real_labels = torch.full(
                (current_batch_size,), 
                label_smooth_real, 
                dtype=torch.float32, 
                device=device
            )
            fake_labels = torch.full(
                (current_batch_size,), 
                label_smooth_fake, 
                dtype=torch.float32, 
                device=device
            )
            
            # ========== Train Discriminator ==========
            optimizer_disc.zero_grad()
            
            # Train on real images
            output_real = discriminator(real_images)
            loss_real = criterion(output_real, real_labels)
            
            # Train on fake images
            noise = generate_noise(current_batch_size, noise_dim, device)
            fake_images = generator(noise).detach()  # Detach to avoid training generator
            output_fake = discriminator(fake_images)
            loss_fake = criterion(output_fake, fake_labels)
            
            # Total discriminator loss
            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            optimizer_disc.step()
            
            running_loss_d += loss_disc.item()
            disc_losses.append(loss_disc.item())
            
            # ========== Train Generator ==========
            # Train generator multiple times per discriminator update if specified
            for _ in range(gen_train_steps):
                optimizer_gen.zero_grad()
                
                # Generate new noise
                noise = generate_noise(current_batch_size, noise_dim, device)
                fake_images = generator(noise)
                
                # Try to fool the discriminator (want it to output high probability)
                output = discriminator(fake_images)
                loss_gen = criterion(output, real_labels)  # Want discriminator to think fakes are real
                
                loss_gen.backward()
                optimizer_gen.step()
                
                running_loss_g += loss_gen.item()
                gen_losses.append(loss_gen.item())
            
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"D_loss: {loss_disc.item():.4f}, G_loss: {loss_gen.item():.4f}")
        
        # Epoch summary
        avg_loss_d = running_loss_d / num_batches
        avg_loss_g = running_loss_g / (num_batches * gen_train_steps)
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Average Discriminator Loss: {avg_loss_d:.4f}")
        print(f"  Average Generator Loss: {avg_loss_g:.4f}")
        print("-" * 60)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                'disc_losses': disc_losses,
                'gen_losses': gen_losses,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Generate and save sample images
            sample_path = os.path.join(save_dir, f"generated_epoch_{epoch+1}.png")
            save_generated_images(
                generator, 
                num_images=64, 
                noise_dim=noise_dim, 
                device=device,
                save_path=sample_path
            )
    
    # Save final model
    final_model_path = os.path.join(save_dir, "final_model.pth")
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'disc_losses': disc_losses,
        'gen_losses': gen_losses,
    }, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Plot and save training losses
    loss_plot_path = os.path.join(save_dir, "training_losses.png")
    plot_training_losses(disc_losses, gen_losses, save_path=loss_plot_path)
    
    # Generate final sample images
    final_sample_path = os.path.join(save_dir, "final_generated_images.png")
    save_generated_images(
        generator, 
        num_images=64, 
        noise_dim=noise_dim, 
        device=device,
        save_path=final_sample_path
    )
    
    print("\nTraining completed!")
    return generator, discriminator, disc_losses, gen_losses


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Train a vanilla GAN on MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--noise-dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--lr-gen', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr-disc', type=float, default=0.0001, help='Discriminator learning rate')
    parser.add_argument('--label-smooth-real', type=float, default=0.9, help='Label smoothing for real images')
    parser.add_argument('--label-smooth-fake', type=float, default=0.1, help='Label smoothing for fake images')
    parser.add_argument('--gen-train-steps', type=int, default=1, help='Generator training steps per discriminator step')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for MNIST data')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        lr_generator=args.lr_gen,
        lr_discriminator=args.lr_disc,
        label_smooth_real=args.label_smooth_real,
        label_smooth_fake=args.label_smooth_fake,
        gen_train_steps=args.gen_train_steps,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


