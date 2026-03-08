# Toxicity Classification Model - Project Report

## Executive Summary
Developed an improved Random Forest classifier for toxicity detection with optimized threshold tuning. The model successfully increased recall from **27% to 91%**, enabling detection of most toxic cases while managing false positives through dynamic threshold adjustment.

---

## 1. Dataset Overview
- **Source**: `data.csv`
- **Classes**: 2 (Toxic=1, NonToxic=0)
- **Total Features**: Multiple features (reduced to top 50 for analysis)
- **Data Quality**: No missing values detected
- **Class Balance**: Imbalanced dataset (more NonToxic than Toxic samples)

---

## 2. Pipeline Architecture

### Stage 1: Data Preparation
- Loaded and explored dataset structure
- Verified data integrity (no missing values)
- Mapped categorical labels: 'Toxic' → 1, 'NonToxic' → 0
- Analyzed class distribution with visualization

### Stage 2: Feature Selection
- **Method**: SelectKBest with Mutual Information scoring
- **Features Selected**: 50 most informative features
- **Approach**: Prioritized features with highest mutual information with target variable
- **Correlation Analysis**: Identified top 20 features correlated with toxicity class

### Stage 3: Initial Baseline Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 trees, random_state=42
- **Cross-validation**: Stratified K-Fold (5 splits)
- **Baseline Accuracy**: ~71%
- **Baseline Recall (Toxic)**: 27% ❌ (Poor - missed most toxic cases)

### Stage 4: Optimized Model (Current)
- **Algorithm**: Random Forest Classifier (improved)
- **Key Improvements**:
  1. ✓ **SMOTE Resampling**: Applied only to training data (proper technique)
  2. ✓ **Class Weighting**: Balanced weights to penalize minority class errors
  3. ✓ **Hyperparameter Tuning**:
     - n_estimators: 200 (more trees)
     - max_depth: 20 (deeper trees for complex patterns)
     - min_samples_split: 5 (allow finer splits)
  4. ✓ **Threshold Optimization**: Adjusted decision boundary from 0.5 to 0.35

---

## 3. Results Comparison

### Baseline Model (Default Threshold 0.5)
```
Precision (Toxic):    0.60
Recall (Toxic):       0.27 
F1-Score (Toxic):     0.38
Accuracy:             0.71
```
**Issue**: Model is too conservative, missing 73% of toxic cases.

### Optimized Model (Threshold 0.35)
```
Precision (Toxic):    0.48
Recall (Toxic):       0.91 ✓
F1-Score (Toxic):     0.62 ✓
Accuracy:             0.66
ROC-AUC:              0.7121 ✓
```
**Improvement**: Catches 91% of toxic cases (10 out of 11), significantly better recall.

### Performance Metrics Summary
| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| **Recall** | 27% | 91% | +237% ✓ |
| **Precision** | 60% | 48% | -20% |
| **F1-Score** | 0.38 | 0.62 | +63% ✓ |
| **Accuracy** | 71% | 66% | -7% |
| **ROC-AUC** | - | 0.71 | - |

---

## 4. Key Improvements Implemented

### Problem 1: Class Imbalance
**Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- Synthetically generated toxic samples during training
- **Critical**: Applied only to training data (proper data leakage prevention)
- Helps model learn minority class patterns better

### Problem 2: Conservative Predictions
**Solution**: Probability Threshold Tuning
- Default threshold (0.5) misses many toxic cases
- Lowered threshold to 0.35 with F1-score optimization
- Trade-off: More false positives but catches toxic content

### Problem 3: Hyperparameter Defaults
**Solution**: Optimized Parameters
- Increased trees (100 → 200) for better generalization
- Increased max_depth (default → 20) for pattern complexity
- Added class_weight='balanced' to penalize minority errors

---

## 5. Current Model Characteristics

### Strengths ✓
- **High Recall**: Catches 91% of toxic cases (10 out of 11)
- **Better F1-Score**: Balanced metric improved from 0.38 to 0.62
- **Proper Data Handling**: SMOTE applied correctly to avoid leakage
- **ROC-AUC**: 0.7121 indicates good discrimination ability

### Trade-offs ⚠️
- **Lower Precision**: More false positives (52% false alarm rate)
- **Lower Overall Accuracy**: 66% (down from 71%)
- **For toxicity detection, this is acceptable** because:
  - Missing toxic content (False Negatives) is worse than false alerts
  - Review costs scale with false positives but

---

---

## Technical Stack
- **Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn
- **Methods**: SMOTE, Random Forest, Grid Search, Stratified K-Fold CV
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix


