# EEG_Lemon_Dataset_preli
This project extracts wavelet-based (DWT) features from segmented EEG signals and trains machine learning models to classify EEG states. It supports EC vs EO (binary) and age/gender (4-class) classification using DNN and classical baselines, with reproducible evaluation scripts and saved fold results.
# EEG DWT Classification (EC/EO + Age/Gender 4-Class)

This repository implements EEG classification using Discrete Wavelet Transform (DWT) features.
It supports:
- **Binary EC vs EO**
- **4-class classification**: young_M, young_F, old_M, old_F (from metadata CSV)

Models included:
- DNN (Keras/TensorFlow)
- SVM (linear)
- Random Forest
- Gradient Boosting

Feature pipeline:
1. Load EEG CSV
2. Segment into fixed-length windows (e.g., 5s or 30s at 250 Hz)
3. Extract DWT features (db4, level=6): per coefficient array compute:
   - standard deviation
   - mean
   - RMS
4. ANOVA F-test feature selection (SelectPercentile)

Evaluation:
- Segment-level Stratified K-Fold CV (fast but can leak subject identity if segments from the same subject appear in both train and test)
- Subject-level holdout split (recommended for realistic generalization)
- Subject-level Group K-Fold CV (recommended)

---

## Setup

### 1) Create environment and install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
