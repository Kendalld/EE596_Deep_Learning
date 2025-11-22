# Key Decisions & User Input Requirements

## 🎯 Critical Architectural Decisions

### Decision 1: MLP vs CNN Architecture

**Choice:** Use **MLP_MO** for masking/noise, **CNN_MO** for de-synchronization

**Why MLP for Masking/Noise:**
- Masking doesn't cause spatial misalignment in traces
- MLP efficiently learns direct power-to-key relationships
- Lower computational cost
- Proven 6-9x speedup over DDLA baseline

**Why CNN for De-Synchronization:**
- De-synchronization randomly shifts traces (0-20 samples)
- **CNN's translation-invariance** handles misaligned patterns
- Convolution + pooling layers recognize patterns at any position
- Proven ~30x speedup over CNN_DDLA baseline

**Implementation Impact:**
- Need to implement both architectures
- MLP_MO: Input → Shared Layer (optional) → 256 branches (20×10-ReLU → 2-Softmax)
- CNN_MO: Input → 2 Conv Blocks → 256 branches

---

### Decision 2: Shared Layer Architecture

**Choice:** Evaluate both **Non-SoSL** (no shared) and **SoSL-200** (200-node shared layer)

**Non-SoSL (No Shared Layer):**
- ✅ Better discrimination (gap: 0.645 vs 0.595)
- ✅ Faster convergence
- ❌ Higher memory (256 independent first layers)

**SoSL-200 (Shared Layer):**
- ✅ Lower memory footprint
- ✅ Better noise robustness (96% vs 80% at σ=1.0)
- ✅ Good performance/efficiency balance
- ❌ Slightly slower convergence

**Paper's Recommendation:** Use SoSL-200 for most cases, Non-SoSL when maximum discrimination needed

---

### Decision 3: Multi-Loss Function

**Choice:** Separate loss per output branch (multi-loss), not single aggregated loss

**Rationale:**
- Each of 256 branches has independent loss: `L[k](θ) = -1/Ns * Σ(y_true * ln(z))`
- Total: `L_total = Σ(γ_k * L[k](θ))` where γ_k = 1
- Enables direct per-key-hypothesis accuracy tracking
- No custom post-processing needed (unlike parallel architectures)
- Lower complexity than binary cross-entropy multilabel approaches

---

## 🔴 Tasks Requiring User Input

### Critical (Blocking Progress)

1. **Dataset Acquisition** ⚠️ **BLOCKER**
   - **Action Required:** Download ASCAD and ChipWhisperer datasets
   - **Location:** 
     - ASCAD: https://github.com/ANSSI-FR/ASCAD
     - ChipWhisperer: Need to locate/download 10K traces, 480 samples/trace
   - **Impact:** Cannot proceed with training without datasets
   - **Alternative:** Use alternative public SCA datasets if originals unavailable

2. **Noise Generation Verification** ⚠️ **VERIFICATION NEEDED**
   - **Action Required:** Verify Gaussian noise parameters match paper
   - **Formula:** `t_noise(i,m) = t(i,m) + σ × randn(1,m) + mean`
   - **Parameters:** σ = 0.5, 1.0, 1.5; mean = 0
   - **Impact:** Incorrect noise may affect success rate comparisons

### Stretch Goal (Optional)

3. **ChipWhisperer Hardware Setup** 🔧 **STRETCH GOAL**
   - **Action Required:** Physical hardware setup and trace collection
   - **Requirements:**
     - ChipWhisperer-lite board
     - Target device with AES implementation
     - Power measurement setup
   - **Impact:** Enables custom trace collection (not required for paper reproduction)

---

## ⚙️ Tasks Agent Can Complete

### Environment & Setup
- ✅ Python environment setup with dependencies
- ✅ Project structure creation
- ✅ Code framework implementation

### Data Processing
- ✅ Multi-output labeling implementation (LSB formula)
- ✅ Dataset reconstruction scripts
- ✅ Noise injection implementation
- ✅ De-synchronization simulation

### Model Implementation
- ✅ MLP_MO architecture (all variants)
- ✅ CNN_MO architecture
- ✅ Multi-loss function implementation
- ✅ Training loop with ADAM optimizer

### Training & Evaluation
- ✅ Training scripts for all experiments
- ✅ Evaluation metrics (accuracy, success rate, attack time)
- ✅ Visualization scripts
- ✅ Results comparison with paper

---

## 📊 Expected Performance Targets

### Masking Countermeasure
- **Attack Time:** 6-9x faster than MLP_DDLA
- **Accuracy Gap:** Correct key ~0.64, Incorrect ~0.60

### Noise Countermeasure
- **σ=1.0:** SoSL-200 should achieve 96% success rate (vs 80% DDLA)
- **σ=1.5:** SoSL-200 should achieve 44% success rate (vs 30% DDLA)
- **σ=1.5 (50K traces):** 100% success rate

### De-Synchronization
- **Attack Time:** ~30x faster than CNN_DDLA
- **Key Detection:** Clear loss/accuracy distinction

---

## 🚦 Progress Checklist

- [ ] **User:** Download ASCAD dataset
- [ ] **User:** Download ChipWhisperer dataset (or confirm alternative)
- [ ] **Agent:** Implement data preprocessing pipeline
- [ ] **Agent:** Implement MLP_MO architecture
- [ ] **Agent:** Implement CNN_MO architecture
- [ ] **Agent:** Implement training scripts
- [ ] **User:** Verify noise generation parameters
- [ ] **Agent:** Run masking experiments
- [ ] **Agent:** Run noise experiments
- [ ] **Agent:** Run de-synchronization experiments
- [ ] **Agent:** Generate visualizations
- [ ] **Agent:** Compare results with paper
- [ ] **Stretch:** ChipWhisperer hardware setup (optional)

