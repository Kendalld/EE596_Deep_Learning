# Final Project: Multi-Output SCA Neural Network

This project reproduces the results from the paper:
**"Efficient Nonprofiled Side-Channel Attack Using Multi-Output Classification Neural Network"**
by Hoang et al. (2023)

## Project Structure

```
labs/Final_project/
├── Final_Project_Kendall_Davies.ipynb  # Main notebook
├── labeling.py                          # LSB labeling and multi-output dataset construction
├── data_loader.py                       # Dataset loading and countermeasure simulation
├── models.py                            # MLPMO and CNNMO model implementations
├── training.py                          # Training loops with multi-loss computation
├── evaluation.py                        # Metrics, plotting, and evaluation utilities
├── datasets/                            # Directory for ASCAD dataset (user must provide)
└── figures/                             # Directory for generated plots and results
```

## Setup

### Prerequisites

1. **Python Environment**: Python 3.11+ with PyTorch 2.2.2+
2. **Required Packages**: 
   - torch, torchvision, torchaudio
   - numpy, matplotlib, seaborn
   - h5py (for loading ASCAD dataset)
   - tqdm (for progress bars)

3. **Install h5py** (if not already installed):
   ```bash
   pip install h5py
   ```

### Dataset

**User Input Required**: You must obtain the ASCAD dataset:

1. Download from: https://github.com/ANSSI-FR/ASCAD
2. Place the HDF5 file in `labs/Final_project/datasets/ASCAD.h5`
3. Update the `ASCAD_PATH` variable in the notebook if using a different location

## Key Components

### Models

- **MLPMO (Multi-Output MLP)**: 
  - Used for masking and noise-generation countermeasures
  - Architecture: Input → Shared Layer (optional) → 256 branches
  - Each branch: 20×10-ReLU → 2-Softmax output

- **CNNMO (Multi-Output CNN)**:
  - Used for de-synchronization countermeasures
  - Architecture: Input → Shared Conv Blocks → 256 branches
  - Leverages translation-invariance for handling shifted traces

### Experiments

1. **Masking Countermeasure**: Compare Non-SoSL vs SoSL-200 vs MLPDDLA
2. **Noise Generation**: Test robustness with σ = 0.5, 1.0, 1.5
3. **De-Synchronization**: Evaluate CNNMO on randomly shifted traces

## Usage

1. Open `Final_Project_Kendall_Davies.ipynb` in Jupyter
2. Run cells sequentially
3. Once ASCAD dataset is available, uncomment and run experiment cells
4. Results will be generated in the `figures/` directory

## Expected Results

- **Attack Time**: 6-9x speedup for MLPMO over MLPDDLA
- **Success Rate**: 20%+ improvement for noisy data (σ = 1.0, 1.5)
- **De-Synchronization**: ~30x speedup for CNNMO over CNNDDLA

## Notes

- The notebook includes placeholder cells that will be populated once the dataset is available
- Baseline implementations (MLPDDLA, CNNDDLA) require user decision on implementation approach
- GPU acceleration is recommended for faster training

