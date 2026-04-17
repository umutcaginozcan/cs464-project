# PyTorch MLP for Stroke Prediction

This module contains the deep learning implementation of the stroke prediction task
using a PyTorch-based Multi-Layer Perceptron (MLP).

The goal is to evaluate whether a neural network can improve performance,
especially on the minority (stroke) class, compared to classical machine learning models.

## 📦 Module Structure

- `mlp_dataset.py`  
  Prepares PyTorch datasets and DataLoaders using preprocessed tabular data.

- `mlp_model.py`  
  Defines the MLP architecture (Linear + ReLU + Dropout layers).

- `losses.py`  
  Implements custom loss functions, including Focal Loss.

- `train_mlp_model.py`  
  Handles training, validation, evaluation, and result export.

- `comparison_analysis.ipynb`  
  Notebook used to compare MLP results with classical ML experiments.

## ⚙️ Workflow

1. Preprocessing is reused from `shared.py`
2. Data is converted to PyTorch tensors
3. MLP model is trained with different loss strategies:
   - BCE
   - Weighted BCE
   - Focal Loss
4. Early stopping is applied based on validation PR-AUC
5. Final model is evaluated on the test set
6. Results are saved as CSV
7. Training history plots are generated
8. Notebook compares deep learning vs classical ML results

## ⚖️ Imbalance Handling

The model supports multiple strategies:

- **BCE (baseline)**
- **Weighted BCE** using `pos_weight`
- **Focal Loss** to emphasize hard/misclassified examples

## 📊 Outputs

Generated outputs include:

- Training history plots:
  - `outputs/figures/dl_mlp_training_history_*.png`
- Test results:
  - `outputs/results/dl_mlp/test_results.csv`

These outputs are designed to be directly comparable with classical model results.

## 📈 Comparison Notebook

The notebook `comparison_analysis.ipynb`:

- loads classical and MLP results
- compares performance across:
  - F1
  - Recall
  - Precision
  - ROC-AUC
  - PR-AUC
- evaluates the effectiveness of imbalance handling strategies

## 🚀 Goal

This module acts as the deep learning counterpart to the classical ML pipeline,
helping assess whether a simple MLP can better capture patterns in tabular clinical data.