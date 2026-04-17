"""
Deep Learning (PyTorch MLP) module for stroke prediction.

This package provides the deep learning pipeline developed for the stroke
prediction project. It implements a multilayer perceptron (MLP) in PyTorch
for binary classification on tabular clinical data, while reusing the shared
preprocessing pipeline from `shared.py` to ensure consistency and fair
comparison with the classical machine learning models.

Main responsibilities of this module:
- loading the preprocessed train/validation/test splits,
- converting tabular features into PyTorch tensors and DataLoaders,
- defining configurable MLP architectures for binary classification,
- training and validating models with different loss functions,
- supporting imbalance-aware learning strategies such as:
  - standard BCEWithLogitsLoss,
  - weighted BCEWithLogitsLoss using `pos_weight`,
  - focal loss,
- tracking training history across epochs,
- applying early stopping based on validation PR-AUC,
- exporting test metrics in a format compatible with classical ML results,
- generating training-history plots for experiment analysis.

The accompanying notebook is used for comparison and analysis of MLP results
against the classical experiments (e.g. baseline, class-weighted, and SMOTE-based
models), making it easier to inspect metrics such as F1, Recall, ROC-AUC,
and PR-AUC across approaches.

This package is intended to support the final experimental comparison and
reporting stages of the project.
"""