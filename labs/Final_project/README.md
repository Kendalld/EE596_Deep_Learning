# Final Project: Reproduction of Multi-Output SCA Neural Network

## Quick Start

This project reproduces the results from the paper:
**"Efficient Nonprofiled Side-Channel Attack Using Multi-Output Classification Neural Network"**
by Van-Phuc Hoang, Ngoc-Tuan Do, and Van Sang Doan (IEEE Embedded Systems Letters, 2023)

## 📋 Documentation

- **[REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md)** - Complete detailed reproduction plan
- **[KEY_DECISIONS.md](KEY_DECISIONS.md)** - Key architectural decisions and rationale
- **This README** - Quick reference and overview

## 🎯 Key Decisions Summary

### MLP vs CNN
- **MLP_MO**: Use for masking and noise countermeasures (6-9x faster than DDLA)
- **CNN_MO**: Use for de-synchronization countermeasures (~30x faster than CNN_DDLA)
- **Rationale**: CNN's translation-invariance handles misaligned traces from de-synchronization

### Shared Layer Architecture
- **Non-SoSL**: Better discrimination, faster convergence, higher memory
- **SoSL-200**: Better noise robustness, lower memory, good balance
- **Paper's Choice**: SoSL-200 for most experiments

## ✅ Dataset Status

1. **ASCAD Dataset** - ✅ **Available**
   - Location: `labs/Final_project/datasets/ASCAD/STM32_AES_v2/ascadv2-extracted.h5`
   - Used for: Masking and noise countermeasure experiments
   - Ready to use in notebook

2. **ChipWhisperer Dataset** - 🔧 **Stretch Goal**
   - Moved to stretch goals (requires hardware setup)
   - Would be used for: De-synchronization countermeasure experiments
   - See stretch goal section below

3. **Noise Parameters** (VERIFICATION)
   - Formula: `t_noise(i,m) = t(i,m) + σ × randn(1,m) + mean`
   - Parameters: σ = 0.5, 1.0, 1.5; mean = 0
   - Implemented and ready to use

## 📊 Expected Results

| Experiment | Metric | Target |
|------------|--------|--------|
| Masking | Speedup | 6-9x faster than DDLA |
| Noise (σ=1.0) | Success Rate | 96% (vs 80% DDLA) |
| Noise (σ=1.5) | Success Rate | 44% (vs 30% DDLA) |
| De-Sync | Speedup | ~30x faster than CNN_DDLA |

## 🚀 Next Steps

1. ✅ **Implementation Complete** - All code has been implemented
2. **Download required datasets** (see above) - **User action required**
3. **Run experiments** using the notebook or experiment scripts
4. Compare results with paper metrics

## 📝 Implementation Status

✅ **Completed:**
- Data preprocessing (multi-output labeling, noise injection, de-sync)
- MLP_MO model (Non-SoSL and SoSL variants)
- CNN_MO model for de-synchronization
- Training utilities and logging
- Evaluation metrics
- Visualization scripts
- Experiment scripts
- Main notebook

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for detailed implementation notes.

## 📁 Project Structure

```
labs/Final_project/
├── README.md (this file)
├── REPRODUCTION_PLAN.md (detailed plan)
├── KEY_DECISIONS.md (architectural decisions)
├── datasets/ (user must populate)
│   ├── ASCAD/
│   └── ChipWhisperer/
├── src/ (to be created by agent)
├── notebooks/ (to be created by agent)
├── figures/ (to be created by agent)
└── results/ (to be created by agent)
```

## 🔧 Stretch Goals

### ChipWhisperer Dataset & De-Synchronization Experiments

**Status:** Moved to stretch goals (not required for main reproduction)

**Requirements:**
- Acquire ChipWhisperer-lite board
- Set up hardware and collect 10,000 traces (480 samples/trace)
- Or locate existing ChipWhisperer dataset

**Experiments:**
- CNN_MO training on de-synchronized data
- Expected: ~30x faster than CNN_DDLA baseline
- See [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md) for details

**Note:** Main experiments (masking and noise) can be completed with ASCAD dataset only.

