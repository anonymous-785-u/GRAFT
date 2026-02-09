# GRAFT: Gated Residual Accelerated Failure Time

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **GRAFT: Decoupling Ranking and Calibration for Survival Analysis**.

## Overview

GRAFT (Gated Residual Accelerated Failure Time) is a novel survival analysis model that addresses key limitations of existing approaches through three main innovations:

1. **Hybrid Architecture**: Combines an interpretable linear AFT model with a non-linear residual neural network
2. **Integrated Feature Selection**: Uses Gaussian-based Stochastic Gates (STG) for automatic, differentiable feature selection
3. **Principled Training**: Employs stochastic conditional imputation from local Kaplan-Meier estimators with C-index-aligned ranking loss

GRAFT outperforms classical and deep learning baselines in both discrimination (C-index) and calibration (IBS), while remaining robust and sparse in high-dimensional, noisy settings.

## Key Features

- **GPU-accelerated** training for all deep learning models (GRAFT, DeepHit, DeepSurv)
- **Local Kaplan-Meier imputation** for handling censored data
- **Global stochastic gates** for population-level feature selection
- **Early stopping** with patience for efficient training
- **Comprehensive evaluation** on 6 benchmark datasets
- **Reproducible** experiments with multiple seeds and cross-validation

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU recommended (optional but significantly faster)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: ~5GB for datasets and outputs

### Python Packages

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Survival analysis
lifelines>=0.27.0
pycox>=0.2.3
scikit-survival>=0.17.0

# Deep learning
torch>=2.0.0
torchsort>=0.1.9

# Utilities
matplotlib>=3.4.0
torchtuples>=0.2.2
```

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/anonymous-785-u/GRAFT.git
cd GRAFT

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn scipy
pip install lifelines pycox scikit-survival
pip install torch torchsort torchtuples
pip install matplotlib
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/anonymous-785-u/GRAFT.git
cd GRAFT

# Create conda environment
conda create -n graft python=3.8
conda activate graft

# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install numpy pandas scikit-learn scipy lifelines pycox scikit-survival torchsort matplotlib torchtuples
```

## Dataset Setup

### Automatic Download (3 datasets)
These datasets are automatically downloaded via `pycox`:
- **GBSG** (German Breast Cancer Study Group)
- **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium)
- **SUPPORT** (Study to Understand Prognoses Preferences Outcomes and Risks of Treatment)

### Manual Download (3 datasets)
Download the following datasets from [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/articles/data.html) and place them in the **same directory** as the experiment scripts:

1. **AIDS** - AIDS Clinical Trial
   - URL: https://vincentarelbundock.github.io/Rdatasets/csv/survival/aids.csv
   - Save as: `aids.csv`

2. **FLCHAIN** - Free Light Chain Study
   - URL: https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv
   - Save as: `flchain_final.csv`

3. **NWTCO** - National Wilms Tumor Study
   - URL: https://vincentarelbundock.github.io/Rdatasets/csv/survival/nwtco.csv
   - Save as: `nwtco.csv`

### Quick Download Commands

```bash
# Download required datasets
wget -O aids.csv https://vincentarelbundock.github.io/Rdatasets/csv/survival/aids.csv
wget -O flchain_final.csv https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv
wget -O nwtco.csv https://vincentarelbundock.github.io/Rdatasets/csv/survival/nwtco.csv
```

**Final directory structure:**
```
GRAFT/
├── Experiment_1.py
├── Experiment_2.py
├── Experiment_3.py
├── aids.csv              # Downloaded
├── flchain_final.csv     # Downloaded
├── nwtco.csv            # Downloaded
├── run_all_experiments.sh
├── LICENSE
└── README.md
```

## Usage

### Experiment 1: Baseline Comparison

Compares GRAFT against 5 baseline models (CoxPH, Weibull, DeepHit, DeepSurv, RSF) on all 6 datasets.

```bash
python Experiment_1.py
```

**What it does:**
- Evaluates all 6 models on 6 datasets
- Uses 3-fold cross-validation with 3 random seeds (42, 43, 44)
- Reports C-index and Integrated Brier Score (IBS)
- Shows both fold-averaged and seed-averaged results

**Expected runtime:** 60-90 minutes (with GPU)

**Outputs:**
- Console tables with detailed results per dataset
- Combined results table across all datasets

**Corresponds to:** Table 1 in the paper

---

### Experiment 2: Ablation Study

Tests the importance of GRAFT's architectural components (STG and Residual MLP) under increasing Gaussian noise.

```bash
python Experiment_2.py
```

**What it does:**
- Tests 3 GRAFT variants:
  - **Full GRAFT**: STG + Residual MLP
  - **No STG**: Only Residual MLP (no feature selection)
  - **Linear Only**: Pure linear AFT (no STG, no MLP)
- Adds Gaussian noise features at 0×, 3×, 5×, 7×, 10× multipliers
- Evaluates on all 6 datasets with 3-fold CV and 3 seeds

**Expected runtime:** 60-90 minutes (with GPU)

**Outputs:**
- Console tables showing performance degradation under noise
- `ablation_cindex_summary.png` - C-index plot (6 subplots, one per dataset)
- `ablation_ibs_summary.png` - IBS plot (6 subplots, one per dataset)

**Corresponds to:** Figure 1 in the paper

**Expected behavior:** Full GRAFT should maintain performance as noise increases, while No STG and Linear Only degrade.

---

### Experiment 3: Noise Robustness

Tests all 6 models' robustness to heavy-tailed noise (Student's t-distribution, df=2).

```bash
python Experiment_3.py
```

**What it does:**
- Compares all 6 models under extreme noise conditions
- Adds Student's t noise (df=2, heavy tails) at 3×, 5×, 7×, 10× multipliers
- Evaluates on all 6 datasets with 3-fold CV and 3 seeds

**Expected runtime:** 90-120 minutes (with GPU)

**Outputs:**
- Console tables showing robustness across noise levels
- `noise_robustness_cindex_summary.png` - C-index plot (6 subplots)
- `noise_robustness_ibs_summary.png` - IBS plot (6 subplots)

**Corresponds to:** Figure 2 in the paper

**Expected behavior:** GRAFT should show the flattest performance curves (best robustness), while models without feature selection (DeepHit, DeepSurv) degrade significantly.

---

### Run All Experiments

To run all three experiments sequentially with automatic logging:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

This will:
- Run all 3 experiments in sequence
- Save outputs to `experiment_1_output.log`, `experiment_2_output.log`, `experiment_3_output.log`
- Display progress and timestamps
- Generate a summary report

**Total expected runtime:** 3-4 hours (with GPU)

## GPU Acceleration

GRAFT, DeepHit, and DeepSurv automatically use GPU if available. To verify GPU usage:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Performance comparison:**
- **With GPU**: ~60-90 minutes per experiment
- **Without GPU (CPU only)**: ~4-6 hours per experiment

## Results Interpretation

### C-Index (Concordance Index)
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Higher is better**
- Measures discrimination (ranking ability)
- GRAFT typically achieves 0.61-0.80 depending on dataset

### IBS (Integrated Brier Score)
- **Range**: 0 (perfect) to 1 (worst)
- **Lower is better**
- Measures calibration (prediction accuracy over time)
- GRAFT typically achieves 0.05-0.20 depending on dataset

### Fold-Averaged vs Seed-Averaged
- **Fold-Averaged**: Mean across folds for each seed, then std across seeds
  - Captures variance from model initialization
- **Seed-Averaged**: Mean across seeds for each fold, then std across folds
  - Captures variance from data partitioning


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Note:** This code is provided for research and educational purposes. For clinical applications, please consult with domain experts and obtain appropriate regulatory approvals.
