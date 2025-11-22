# Vanilla GAN for MNIST

A clean, educational implementation of a vanilla Generative Adversarial Network (GAN) for generating MNIST handwritten digits using PyTorch.

## Overview

This implementation includes:
- **Generator**: Fully connected neural network that transforms random noise into 28x28 grayscale images
- **Discriminator**: Fully connected neural network that classifies images as real or fake
- **Training Script**: Complete training loop with checkpointing and visualization
- **Utilities**: Helper functions for data loading, visualization, and model management

## Architecture

### Generator
- Input: Random noise vector (default: 100 dimensions)
- Architecture: Fully connected layers [256 → 512 → 1024 → 784]
- Activations: ReLU for hidden layers, Tanh for output (range [-1, 1])
- Output: 28×28 grayscale images

### Discriminator
- Input: 28×28 grayscale images (flattened to 784)
- Architecture: Fully connected layers [1024 → 512 → 256 → 1]
- Activations: LeakyReLU (slope=0.2) for hidden layers, Sigmoid for output
- Regularization: Dropout (0.3) after each hidden layer
- Output: Probability score (0 = fake, 1 = real)

## Usage

### Basic Training

```bash
# From the vanilla_gan_mnist directory
python train.py
```

### Custom Training Parameters

```bash
python train.py \
    --epochs 100 \
    --batch-size 128 \
    --noise-dim 100 \
    --lr-gen 0.0002 \
    --lr-disc 0.0001 \
    --save-dir ./my_checkpoints
```

### Command-Line Arguments

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 128)
- `--noise-dim`: Dimension of noise vector (default: 100)
- `--lr-gen`: Generator learning rate (default: 0.0002)
- `--lr-disc`: Discriminator learning rate (default: 0.0001)
- `--label-smooth-real`: Label smoothing for real images (default: 0.9)
- `--label-smooth-fake`: Label smoothing for fake images (default: 0.1)
- `--gen-train-steps`: Generator training steps per discriminator step (default: 1)
- `--data-dir`: Directory for MNIST data (default: ./data)
- `--save-dir`: Directory to save checkpoints (default: ./checkpoints)
- `--seed`: Random seed for reproducibility (default: 42)

## File Structure

```
vanilla_gan_mnist/
├── models.py          # Generator and Discriminator class definitions
├── train.py           # Main training script
├── utils.py           # Utility functions (data loading, visualization, etc.)
└── README.md          # This file
```

## Outputs

During training, the script will save:
- **Checkpoints**: Model state dictionaries every 10 epochs (`checkpoint_epoch_N.pth`)
- **Generated Images**: Sample images every 10 epochs (`generated_epoch_N.png`)
- **Training Losses**: Plot of discriminator and generator losses (`training_losses.png`)
- **Final Model**: Complete model after training (`final_model.pth`)
- **Final Images**: Final generated samples (`final_generated_images.png`)

## Training Tips

1. **Label Smoothing**: Helps prevent the discriminator from becoming overconfident
   - Real labels: 0.9 instead of 1.0
   - Fake labels: 0.1 instead of 0.0

2. **Learning Rates**: Discriminator typically uses a lower learning rate to prevent it from becoming too strong too quickly

3. **Generator Training Steps**: Training the generator multiple times per discriminator update can help balance training

4. **Monitoring**: Watch the loss curves - both losses should decrease over time. If discriminator loss drops to near zero, the generator may need more training steps.

## Requirements

- PyTorch 2.2.2+
- torchvision 0.17.2+
- matplotlib
- numpy

## Example Usage in Python

```python
from models import Generator, Discriminator
from utils import get_device, save_generated_images
import torch

# Load trained model
device = get_device()
generator = Generator(input_noise_dim=100, image_size=784)
checkpoint = torch.load('checkpoints/final_model.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.to(device)
generator.eval()

# Generate images
save_generated_images(generator, num_images=64, noise_dim=100, device=device)
```

## Notes

- The model uses normalized images in the range [-1, 1] to match the Tanh output of the generator
- Images are automatically denormalized for visualization
- Training is deterministic when using the same seed
- GPU acceleration is automatically used if available


