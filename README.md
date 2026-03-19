# Hybrid ML Reentrancy Detector for Solidity Smart Contracts

A machine learning system that combines **Slither static analysis** with **source-level code features** to detect reentrancy vulnerabilities in Solidity smart contracts. Achieves **88.2% accuracy** and **87.6% F1-score**, significantly outperforming Slither alone.

## Results

### Model Comparison (5-Fold Stratified CV, 365 contracts)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | **88.22%** | **83.65%** | **92.14%** | **87.61%** |
| Random Forest | 86.30% | 81.39% | 91.00% | 85.81% |
| SVM (RBF) | 86.03% | 79.42% | 94.56% | 86.12% |
| Logistic Regression | 79.73% | 77.51% | 79.54% | 78.00% |

### Hybrid ML vs Slither Alone

| Metric | Slither Only | Hybrid ML (RF) | Improvement |
|---|---|---|---|
| Recall | 73.1% | 97.6% | **+24.5%** |
| Precision | 56.7% | 87.2% | **+30.5%** |
| TP | 122 | 163 | +41 |
| FN | 45 | 4 | -41 |
| FP | 93 | 24 | -69 |

> The hybrid model catches **41 more vulnerable contracts** that Slither misses, while reducing false positives by 69.

## Features

The model uses **17 features** extracted from two sources:

**Slither-derived features (7):**
- `has_reentrancy_eth` — Slither reentrancy-eth detector triggered
- `num_reentrancy_detectors` — Number of reentrancy detectors triggered
- `cei_violation` — Check-Effect-Interaction violation detected (from CFG analysis)
- `has_dangerous_call` — Dangerous external call patterns
- `has_low_level_call` — Low-level `.call()` usage
- `num_external_calls` — Count of external calls
- `has_state_set_before_call` — State written before external call

**Source code features (10):**
- `num_functions` — Number of functions in the contract
- `has_modifier` — Uses function modifiers
- `has_reentrancy_guard` — Has reentrancy guard pattern (e.g., OpenZeppelin's ReentrancyGuard)
- `has_emit_after_call` — Event emitted after external call
- `has_mutex_pattern` — Manual mutex lock pattern
- `num_require_assert` — Number of require/assert statements
- `num_modifiers_applied` — Total modifiers applied to functions
- `has_internal_ext_call` — Internal function making external call
- `has_interface_cast` — Interface casting pattern
- `lines_of_code` — Contract size

## Project Structure

```
reentrancy_ml_project/
├── src/
│   ├── train_and_visualize.py    # Train 4 models + generate figures
│   ├── extract_features_v2.py    # Extract 17 features from dataset
│   ├── baseline_slither.py       # Slither-only baseline evaluation
│   ├── config.py                 # Paths and constants
│   ├── dataset.py                # Dataset loader
│   └── slither_utils.py          # Slither runner utilities
├── webapp/
│   ├── app.py                    # Flask web demo
│   ├── .env                      # GROQ_API_KEY (create this)
│   ├── model/                    # Trained model artifacts
│   │   ├── rf_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_cols.pkl
│   └── templates/
│       └── index.html            # Web UI
├── outputs/
│   ├── features_v2_all.csv       # Full dataset (365 contracts, 17 features)
│   ├── features_v2_04.csv        # Solidity 0.4.x contracts
│   ├── features_v2_05.csv        # Solidity 0.5.x contracts
│   ├── features_v2_08.csv        # Solidity 0.8.x contracts
│   ├── baseline_slither.csv      # Slither baseline results
│   ├── slither_*_results.csv     # Raw Slither output
│   └── figures_v2/               # Generated plots
│       ├── confusion_matrices.png
│       ├── metrics_comparison.png
│       └── roc_curves.png
└── README.md
```

## Dataset

- **365 manually verified Solidity contracts** (167 reentrant, 198 safe)
- Covers Solidity versions **0.4.x, 0.5.x, and 0.8.x**
- Each contract is labeled by human experts as reentrant or safe

The training dataset is sourced from [matteo-rizzo/manually-verified-reentrancy-dataset](https://github.com/matteo-rizzo/manually-verified-reentrancy-dataset). We gratefully acknowledge the authors for creating and publicly sharing this high-quality, manually verified benchmark for reentrancy vulnerability detection.

> M. Rizzo et al., "manually-verified-reentrancy-dataset" — A curated dataset of Solidity smart contracts with expert-labeled reentrancy annotations.

## Setup

### Prerequisites

- Python 3.10+
- [Slither](https://github.com/crytic/slither) (static analyzer)
- [solc-select](https://github.com/crytic/solc-select) (Solidity compiler manager)

### Installation

```bash
# Clone the repo
git clone https://github.com/Pulchi/ML_Reentrancy_Detector.git
cd ML_Reentrancy_Detector

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy scikit-learn xgboost matplotlib joblib flask groq

# Install Slither and solc-select
pip install slither-analyzer solc-select
solc-select install 0.4.24 0.5.0 0.8.0
```

### Running the Web Demo

```bash
cd webapp
python app.py
```

Open http://127.0.0.1:5000 in your browser. Upload a `.sol` file to analyze.

#### Optional: Enable LLM Analysis

The demo supports AI-powered explanations via Groq (LLaMA 3.3 70B):

1. Get a free API key at https://console.groq.com
2. Create `webapp/.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Restart the app

### Training Models

```bash
python src/train_and_visualize.py
```

This trains Random Forest, XGBoost, Logistic Regression, and SVM with 5-fold stratified cross-validation and generates figures in `outputs/figures_v2/`.

### Extracting Features

To re-extract features from the dataset:

```bash
python src/extract_features_v2.py
```

Requires `slither` and `solc-select` to be installed.

## How It Works

1. **Upload** a Solidity contract (`.sol` file)
2. **Slither Analysis** — Runs Slither to detect reentrancy patterns, extract CFG, and analyze call flow
3. **Feature Extraction** — Extracts 17 features: 7 from Slither output + 10 from source code
4. **ML Prediction** — Random Forest classifier predicts reentrancy probability
5. **Explanation** — Rule-based explanation + optional LLM-powered analysis

The key innovation is the **CEI (Check-Effect-Interaction) violation detection** from Slither's Control Flow Graph (CFG), which identifies state writes after external calls — the root cause of reentrancy.

## Cross-Validation Results

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 5-Fold CV | 86.30% | 81.39% | 91.00% | 85.81% |
| 10-Fold CV | 87.09% | 82.92% | 91.65% | 86.75% |
| Leave-One-Out | 87.95% | 84.36% | 90.42% | 87.28% |

## Top Feature Importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | lines_of_code | 18.1% |
| 2 | cei_violation | 16.5% |
| 3 | num_modifiers_applied | 9.7% |
| 4 | num_functions | 9.0% |
| 5 | num_reentrancy_detectors | 8.9% |

## License

MIT
