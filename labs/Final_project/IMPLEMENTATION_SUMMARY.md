# Implementation Summary

## Overview

This document summarizes the implementation of the multi-output side-channel attack reproduction project.

## Completed Components

### 1. Data Preprocessing (`src/data_preprocessing.py`)
- ✅ Multi-output labeling: `create_multi_output_labels()` - Creates 256 labels per trace (one per key hypothesis)
- ✅ AES S-box implementation for LSB computation
- ✅ Gaussian noise injection: `inject_gaussian_noise()` - For noise robustness experiments
- ✅ De-synchronization: `apply_desynchronization()` - Random trace shifting for de-sync experiments
- ✅ Trace normalization: `normalize_traces()` - Standard, minmax, and L2 normalization
- ✅ Dataset loaders: `load_ascad_dataset()`, `load_chipwhisperer_dataset()` - Placeholder implementations

### 2. MLP_MO Model (`src/mlp_mo.py`)
- ✅ MLP_MO architecture with configurable shared layer
  - Non-SoSL variant (shared_layer_size=0)
  - SoSL variants (shared_layer_size=50, 200, 400)
  - 256 parallel branches, each with 2 hidden layers (20→10 nodes) and 2-class output
- ✅ Multi-output loss function: `MultiOutputLoss()` - Separate loss per branch
- ✅ Branch accuracy computation: `compute_branch_accuracy()` - Per-key-hypothesis accuracy

### 3. CNN_MO Model (`src/cnn_mo.py`)
- ✅ CNN_MO architecture for de-synchronization
  - Two convolutional blocks with BatchNorm and AveragePooling
  - 256 parallel branches processing flattened features
  - Translation-invariant design for misaligned traces

### 4. Training Utilities (`src/training.py`)
- ✅ `PowerTraceDataset` - PyTorch Dataset class for power traces
- ✅ `TrainingLogger` - Logs training metrics to JSON
- ✅ `train_mlp_mo()` - Training function for MLP_MO
- ✅ `train_cnn_mo()` - Training function for CNN_MO
- ✅ Tracks: loss, accuracy, branch accuracies, attack time, correct/incorrect key metrics

### 5. Evaluation Metrics (`src/evaluation.py`)
- ✅ `evaluate_model()` - Comprehensive model evaluation
- ✅ `compute_success_rate()` - Success rate from repeated attacks
- ✅ `compute_key_ranking()` - Ranking of correct key among hypotheses
- ✅ `run_repeated_attacks()` - Run multiple attacks for statistical significance
- ✅ `compare_with_baseline()` - Compare with DDLA baseline

### 6. Visualization (`src/visualization.py`)
- ✅ `plot_accuracy_curves()` - Training accuracy over epochs
- ✅ `plot_attack_time_comparison()` - Bar chart of attack times
- ✅ `plot_success_rate_comparison()` - Success rate comparisons
- ✅ `plot_branch_accuracies()` - 256-branch accuracy visualization
- ✅ `plot_noise_comparison()` - Success rate vs noise level
- ✅ `plot_desync_comparison()` - De-synchronization attack comparison

### 7. Experiment Scripts (`src/experiments.py`)
- ✅ `experiment_masking_mlp_mo()` - Masking countermeasure experiment
- ✅ `experiment_noise_robustness()` - Noise robustness evaluation
- ✅ `experiment_desync_cnn_mo()` - De-synchronization experiment

### 8. Notebook (`notebooks/Final_Project_Reproduction.ipynb`)
- ✅ Complete notebook framework with all experiments
- ✅ Ready-to-run code (uncomment when datasets available)
- ✅ Includes all three main experiments

## Project Structure

```
labs/Final_project/
├── README.md                          # Project overview
├── REPRODUCTION_PLAN.md               # Detailed reproduction plan
├── KEY_DECISIONS.md                   # Architectural decisions
├── IMPLEMENTATION_SUMMARY.md          # This file
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_preprocessing.py          # Data loading and preprocessing
│   ├── mlp_mo.py                      # MLP_MO model and loss
│   ├── cnn_mo.py                      # CNN_MO model
│   ├── training.py                    # Training utilities
│   ├── evaluation.py                  # Evaluation metrics
│   ├── visualization.py               # Plotting functions
│   └── experiments.py                 # Experiment scripts
├── notebooks/
│   └── Final_Project_Reproduction.ipynb  # Main notebook
├── datasets/                          # (User must populate)
│   ├── ASCAD/
│   └── ChipWhisperer/
├── figures/                           # Generated plots
└── results/                           # Training logs and metrics
    └── training_logs/
```

## Usage Instructions

### 1. Setup Environment

```bash
cd /home/kdavies/gitcode/ee596-deep-learning
source .venv/bin/activate  # or: uv venv && source .venv/bin/activate
uv sync
```

### 2. Download Datasets

**Required:**
- ASCAD dataset: Download from https://github.com/ANSSI-FR/ASCAD
  - Store in: `labs/Final_project/datasets/ASCAD/`
- ChipWhisperer dataset: 10,000 traces, 480 samples/trace
  - Store in: `labs/Final_project/datasets/ChipWhisperer/`

### 3. Run Experiments

**Option A: Using the Notebook**
```bash
uv run jupyter lab
# Open: notebooks/Final_Project_Reproduction.ipynb
# Uncomment experiment cells and run
```

**Option B: Using Experiment Scripts**
```python
from src.experiments import experiment_masking_mlp_mo

# Run masking experiment
experiment_masking_mlp_mo(
    dataset_path="datasets/ASCAD",
    dataset_name="Dataset2",
    shared_layer_size=200,
    num_epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

## Key Features

### Multi-Output Architecture
- Single training process predicts all 256 key hypotheses
- 6-9x faster than DDLA baseline (MLP_MO)
- ~30x faster than CNN_DDLA baseline (CNN_MO)

### Shared Layer Variants
- **Non-SoSL**: Better discrimination, faster convergence
- **SoSL-200**: Better noise robustness, lower memory

### Noise Robustness
- Evaluates performance at σ = 0.5, 1.0, 1.5
- Target: +20% success rate improvement at high noise

### De-Synchronization Handling
- CNN architecture with translation-invariance
- Handles random shifts up to 20 samples

## Expected Results

### Masking Countermeasure
- Attack time: 6-9x faster than MLP_DDLA
- Accuracy gap: Correct key ~0.64, Incorrect ~0.60

### Noise Countermeasure
- σ=1.0: 96% success rate (vs 80% DDLA)
- σ=1.5: 44% success rate (vs 30% DDLA)

### De-Synchronization
- Attack time: ~30x faster than CNN_DDLA
- Clear key detection from loss/accuracy metrics

## Implementation Notes

1. **Relative Imports**: The code uses relative imports (`.module`). When running scripts directly, use the notebook or run as a module:
   ```bash
   python -m src.experiments
   ```

2. **Dataset Format**: The dataset loaders expect:
   - ASCAD: HDF5 format (`.h5` files)
   - ChipWhisperer: NumPy format (`.npz` files)
   - Adjust loaders if your datasets use different formats

3. **Device Support**: Code automatically detects CUDA availability. Set `device='cpu'` to force CPU usage.

4. **Random Seeds**: Set seeds at the start of experiments for reproducibility:
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   ```

## Next Steps

1. **Download Datasets** (User action required)
   - ASCAD: https://github.com/ANSSI-FR/ASCAD
   - ChipWhisperer: Locate and download 10K traces

2. **Verify Dataset Formats**
   - Check if datasets match expected HDF5/NumPy formats
   - Adjust loaders if needed

3. **Run Experiments**
   - Start with Experiment 1 (Masking)
   - Then Experiment 2 (Noise)
   - Finally Experiment 3 (De-sync)

4. **Compare Results**
   - Compare attack times with paper
   - Verify success rates match expected values
   - Generate visualizations

## Troubleshooting

### Import Errors
If you see `ImportError: attempted relative import`, use the notebook or run as a module:
```bash
cd labs/Final_project
python -m src.experiments
```

### Dataset Not Found
- Verify dataset paths in experiment scripts
- Check dataset file names match expected format
- Ensure datasets are in correct directories

### CUDA Out of Memory
- Reduce batch size
- Use CPU: `device='cpu'`
- Reduce number of epochs for testing

## References

- Paper: "Efficient Nonprofiled Side-Channel Attack Using Multi-Output Classification Neural Network"
- ASCAD: https://github.com/ANSSI-FR/ASCAD
- PyTorch Documentation: https://pytorch.org/docs/





