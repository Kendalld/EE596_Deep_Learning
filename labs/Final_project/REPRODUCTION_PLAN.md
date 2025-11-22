# Reproduction Plan: Efficient Nonprofiled Side-Channel Attack Using Multi-Output Classification Neural Network

## Paper Summary

This paper introduces a nonprofiled side-channel attack (SCA) technique using multi-output classification neural networks. The key innovation is predicting all 256 key hypotheses in a single training process, rather than requiring repeated training like the baseline DDLA (Differential Deep Learning Analysis) method.

**Key Contributions:**
- MLP_MO: Multi-output MLP for masking and noise countermeasures (6-9x faster than DDLA)
- CNN_MO: Multi-output CNN for de-synchronization countermeasures (~30x faster than CNN_DDLA)
- Improved success rates on noisy data (+20% at σ=1.0, 1.5)

## Key Architectural Decisions

### 1. MLP vs CNN Selection

**Decision:** Use MLP_MO for masking/noise countermeasures, CNN_MO for de-synchronization.

**Rationale:**
- **MLP_MO** is chosen for masking and noise because:
  - Masking countermeasures don't introduce spatial misalignment
  - MLP can effectively learn the relationship between power traces and key-dependent intermediate values
  - Simpler architecture with lower computational overhead
  - The paper shows MLP_MO achieves 6-9x speedup over MLP_DDLA

- **CNN_MO** is chosen for de-synchronization because:
  - De-synchronization introduces random shifts in traces (up to 20 samples)
  - CNNs have **translation-invariance** property through convolution and pooling layers
  - This allows the model to recognize patterns regardless of their position in the trace
  - The paper demonstrates CNN_MO achieves ~30x speedup over CNN_DDLA on de-synchronized traces

### 2. Shared Layer vs Non-Shared Architecture

**Decision:** Evaluate both Non-SoSL (no shared layer) and SoSL-200 (shared layer with 200 nodes).

**Rationale:**
- **Non-SoSL** (no shared layer):
  - Each of 256 branches has independent first hidden layer
  - Better discrimination between correct/incorrect keys (clearer gap: 0.645 vs 0.595)
  - Faster convergence (discriminates correct key earlier)
  - Higher memory usage (256 independent first layers)

- **SoSL-200** (shared layer with 200 nodes):
  - Shared first layer reduces memory footprint
  - Good balance between performance and efficiency
  - Slightly slower convergence but still 7-9x faster than DDLA
  - Better noise robustness (96% success rate at σ=1.0 vs 80% for DDLA)

**Paper's Choice:** SoSL-200 for most experiments (good balance), Non-SoSL for best discrimination.

### 3. Multi-Loss vs Single Loss

**Decision:** Use separate loss function per output branch (multi-loss).

**Rationale:**
- Each of 256 branches has its own loss: L[k](θ) = -1/Ns * Σ(y_true * ln(z))
- Total loss: L_total = Σ(γ_k * L[k](θ)) where γ_k = 1 for all branches
- Enables direct monitoring of accuracy per key hypothesis during training
- No need for custom post-processing function (unlike parallel architecture approaches)
- Lower computational complexity than multilabel approaches using binary cross-entropy

## Dataset Requirements

### ASCAD Dataset (Masking Countermeasure)
- **Dataset1, Dataset2, Dataset3**: Different sizes of ASCAD fixed-key dataset
- Leakage model: Output of third Sbox with unknown mask values
- Used for: Masking countermeasure evaluation, noise generation experiments

### ChipWhisperer-lite Dataset
- **Dataset4, Dataset5**: 10,000 power traces, 480 samples/trace
- Leakage model: Power consumption of first Sbox output process
- Used for: De-synchronization countermeasure evaluation

### Derived Datasets
- **DatasetX-N1, DatasetX-N2, DatasetX-N3**: ASCAD with Gaussian noise (σ=0.5, 1.0, 1.5)
- **Dataset4-sh20, Dataset5-sh20**: ChipWhisperer traces with random shifts (max 20 samples)

## Implementation Plan

### Phase 1: Environment Setup & Data Acquisition

#### Task 1.1: Environment Setup ⚙️ (Agent can do)
- Set up Python 3.13 environment with PyTorch 2.8+ or TensorFlow/Keras
- Install dependencies: NumPy, Matplotlib, Seaborn, scikit-learn
- Verify GPU availability if using CUDA

#### Task 1.2: Dataset Acquisition 🔴 (Requires User Input)
**Status:** User must download datasets
- **ASCAD dataset**: Download from [ASCAD repository](https://github.com/ANSSI-FR/ASCAD)
  - Need: Fixed key dataset (masking countermeasure)
  - Location: Store in `labs/Final_project/datasets/ASCAD/`
- **ChipWhisperer dataset**: Download from ChipWhisperer resources
  - Need: 10,000 traces, 480 samples/trace
  - Location: Store in `labs/Final_project/datasets/ChipWhisperer/`
- **Note:** If datasets are not publicly available, user may need to:
  - Contact paper authors
  - Use alternative public SCA datasets
  - Generate synthetic data following paper methodology

### Phase 2: Data Preprocessing

#### Task 2.1: Implement Multi-Output Labeling ⚙️ (Agent can do)
- Implement LSB labeling formula: `l_i^j = LSB(Sbox(p_i ⊕ k_j))`
  - `p_i`: i-th plaintext (i = 1 to n)
  - `k_j`: j-th key guess (j = 0 to 255)
  - `Sbox`: AES S-box function
- Create multi-output dataset structure:
  - Input: Power traces (n traces × m samples)
  - Output: 256 labels per trace (one per key hypothesis)
  - Format: (n, m) traces, (n, 256) labels

#### Task 2.2: Dataset Reconstruction ⚙️ (Agent can do)
- Reconstruct Dataset1, Dataset2, Dataset3 from ASCAD
- Reconstruct Dataset4, Dataset5 from ChipWhisperer data
- Create noisy variants: DatasetX-N1, DatasetX-N2, DatasetX-N3
  - Formula: `t_noise(i,m) = t(i,m) + σ × randn(1,m) + mean`
  - σ values: 0.5, 1.0, 1.5
- Create de-synchronized variants: Dataset4-sh20, Dataset5-sh20
  - Randomly shift each trace by 0-20 samples

### Phase 3: Model Implementation

#### Task 3.1: Implement MLP_MO Architecture ⚙️ (Agent can do)
**Architecture:**
```
Input Layer: (trace_length,)
  ↓
Shared Layer (optional): 0, 50, 200, or 400 nodes
  ↓
256 Branches (parallel):
  Each branch:
    Hidden Layer 1: 20 nodes, ReLU
    Hidden Layer 2: 10 nodes, ReLU
    Output Layer: 2 nodes, Softmax
```

**Key Implementation Details:**
- Support both Non-SoSL (shared_layer_size=0) and SoSL variants
- Use same architecture as MLP_DDLA for branches (except input layer)
- Initialize weights equivalently for all branches

#### Task 3.2: Implement CNN_MO Architecture ⚙️ (Agent can do)
**Architecture:**
```
Input Layer: (trace_length, 1)
  ↓
Shared Layers:
  Block 1: Conv1D → BatchNorm → AvgPool → ReLU
  Block 2: Conv1D → BatchNorm → AvgPool → ReLU
  ↓
256 Branches (parallel):
  Each branch: (same as CNN_DDLA except output layer)
```

**Key Implementation Details:**
- Use 1D convolutions for time-series power traces
- Batch normalization for training stability
- Average pooling for translation invariance

#### Task 3.3: Implement Multi-Loss Function ⚙️ (Agent can do)
- Implement per-branch loss: `L[k](θ) = -1/Ns * Σ(y_true * ln(z))`
- Implement total loss: `L_total = Σ(γ_k * L[k](θ))` where γ_k = 1
- Ensure separate loss/accuracy tracking per key hypothesis
- Use ADAM optimizer with default settings

### Phase 4: Training & Evaluation

#### Task 4.1: Train MLP_MO on Masking Datasets ⚙️ (Agent can do)
- Train Non-SoSL and SoSL-200 on Dataset2
- Compare with MLP_DDLA baseline (if available) or implement baseline
- Metrics to track:
  - Accuracy per key hypothesis per epoch
  - Attack time (total training time)
  - Gap between correct and incorrect key accuracies
- Target: 6-9x faster than DDLA

#### Task 4.2: Evaluate on Noisy Data 🔴 (Requires User Input)
**Status:** User must verify noise generation parameters
- Train on DatasetX-N1, DatasetX-N2, DatasetX-N3
- Run 50 repeated attacks for statistical significance
- Calculate success rate: percentage of successful attacks
- Compare: MLP_DDLA vs Non-SoSL vs SoSL-200
- Target: +20% success rate improvement at σ=1.0, 1.5

#### Task 4.3: Train CNN_MO on De-Synchronized Data ⚙️ (Agent can do)
- Train CNN_MO on Dataset4-sh20, Dataset5-sh20
- Compare with CNN_DDLA baseline (if available) or implement baseline
- Use loss metric to identify correct key
- Target: ~30x faster than CNN_DDLA

#### Task 4.4: Implement Evaluation Metrics ⚙️ (Agent can do)
- Accuracy per key hypothesis (per epoch)
- Success rate calculation (for repeated attacks)
- Attack time measurement (wall-clock time)
- Key ranking (identify correct key position)

### Phase 5: Results Reproduction & Visualization

#### Task 5.1: Reproduce Masking Results ⚙️ (Agent can do)
- Generate accuracy curves (similar to Fig. 2a, 2b)
- Generate attack time comparison (similar to Fig. 2c)
- Verify: Non-SoSL and SoSL-200 are 6-9x faster than DDLA

#### Task 5.2: Reproduce Noise Results ⚙️ (Agent can do)
- Generate success rate comparison (similar to Fig. 3a)
- Generate accuracy curves for high noise (similar to Fig. 3b, 3c)
- Verify: +20% improvement at σ=1.0, 1.5

#### Task 5.3: Reproduce De-Synchronization Results ⚙️ (Agent can do)
- Generate attack comparison plots (similar to Fig. 4)
- Compare: CPA vs CNN_DDLA vs CNN_MO
- Verify: ~30x speedup over CNN_DDLA

#### Task 5.4: Create Visualization Scripts ⚙️ (Agent can do)
- Plot accuracy curves per key hypothesis
- Plot success rate comparisons
- Plot attack time comparisons
- Save figures in `labs/Final_project/figures/`

## Stretch Goal: ChipWhisperer Trace Collection

### Task SG.1: ChipWhisperer Hardware Setup 🔴 (Requires User Input)
**Status:** Requires physical hardware and user expertise
- Acquire ChipWhisperer-lite board
- Set up hardware connections
- Install ChipWhisperer software and drivers
- Verify board communication

### Task SG.2: Trace Collection Procedure 🔴 (Requires User Input)
**Status:** Requires user to perform hardware operations
- Configure target device (AES implementation)
- Set up power measurement
- Collect 10,000 power traces
- Verify trace quality (480 samples/trace)
- Save traces in appropriate format

### Task SG.3: Process Collected Traces ⚙️ (Agent can do)
- Load collected traces
- Apply preprocessing (normalization, alignment)
- Integrate into existing pipeline
- Run MLP_MO/CNN_MO attacks

## Tasks Requiring User Input

The following tasks **cannot be automated** and require user action:

1. **Dataset Acquisition** (Task 1.2)
   - Download ASCAD and ChipWhisperer datasets
   - Verify dataset availability and format
   - Handle dataset access restrictions if any

2. **Noise Generation Verification** (Task 4.2)
   - Verify Gaussian noise parameters match paper
   - Confirm noise distribution matches paper methodology

3. **Hardware Setup** (Stretch Goal - Task SG.1, SG.2)
   - Physical ChipWhisperer board setup
   - Hardware configuration and trace collection

## Expected Results

### Masking Countermeasure
- **Attack Time:** 6-9x faster than MLP_DDLA
  - Non-SoSL: ~6x faster
  - SoSL-200: ~7-9x faster
- **Accuracy Gap:** Clear distinction between correct (0.636-0.645) and incorrect (0.595-0.609) keys

### Noise Countermeasure
- **Success Rate at σ=0.5:** 100% (all models)
- **Success Rate at σ=1.0:** 
  - MLP_DDLA: 80%
  - SoSL-200: 96% (+16%)
- **Success Rate at σ=1.5:**
  - MLP_DDLA: 30%
  - SoSL-200: 44% (+14%)
  - With 50,000 traces: 100%

### De-Synchronization Countermeasure
- **Attack Time:** ~30x faster than CNN_DDLA
  - CNN_DDLA: ~20,792 seconds
  - CNN_MO: ~704 seconds
- **Key Detection:** Clear distinction in loss/accuracy metrics

## File Structure

```
labs/Final_project/
├── REPRODUCTION_PLAN.md (this file)
├── datasets/
│   ├── ASCAD/
│   │   ├── Dataset1/
│   │   ├── Dataset2/
│   │   └── Dataset3/
│   └── ChipWhisperer/
│       ├── Dataset4/
│       └── Dataset5/
├── src/
│   ├── data_preprocessing.py
│   ├── mlp_mo.py
│   ├── cnn_mo.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
├── notebooks/
│   └── Final_Project_Reproduction.ipynb
├── figures/
│   ├── masking_accuracy.png
│   ├── masking_time.png
│   ├── noise_success_rate.png
│   └── desync_comparison.png
└── results/
    ├── training_logs/
    └── metrics.json
```

## Dependencies

- Python 3.13
- PyTorch 2.8+ or TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- (Optional) CUDA for GPU acceleration

## Notes

- The paper uses Keras framework, but PyTorch can be used as alternative
- Hyperparameters may need tuning based on available hardware
- Baseline implementations (MLP_DDLA, CNN_DDLA) may need to be implemented if not available
- Some datasets may require special access or alternative sources

