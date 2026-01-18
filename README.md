# itbnb — Threshold-Based Naive Bayes

Threshold-Based Naïve Bayes (Tb-NB) is a reformulation of the classical
Naïve Bayes classifier for **binary classification**, where the final
decision is taken by comparing a continuous score against a
**data-driven decision threshold** instead of relying on posterior probabilities.
An optional iterative refinement procedure (iTb-NB) can be enabled to
adapt the decision boundary locally around uncertain regions.
The implementation follows the scikit-learn API and is designed to be
used as a drop-in classifier in standard pipelines.

---

## Overview

In classical Naïve Bayes, posterior probabilities are often poorly calibrated
due to the violation of the conditional independence assumption. While these probabilities are useful for ranking observations, their direct
interpretation are often unreliable.

Tb-NB addresses this issue by:

- computing a continuous log-likelihood ratio score;
- learning an optimal decision threshold τ directly from the data via
  cross-validation;
- optionally refining the threshold locally using an iterative procedure.

---

## Main features

- Binary Naïve Bayes classifier with threshold-based decision rule
- Cross-validated threshold optimization via `ThresholdOptimizer`
- Multiple optimization criteria (accuracy, F1, MCC, balanced error, etc.)
- Optional iterative refinement near the decision boundary (iTb-NB)
- Fully compatible with scikit-learn (`fit`, `predict`, `decision_function`)
- Works with dense or sparse (CSR) feature matrices
- Designed for use in pipelines and model selection workflows

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/francescotiddia/itbnb.git
cd itbnb
pip install -e .
````

---

## Basic usage

```python
import numpy as np
from scipy.sparse import csr_matrix
from itbnb import TbNB

# Example binary Bag-of-Words matrix
X = csr_matrix([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
])
y = np.array([1, 0, 1, 0])

# Initialize Tb-NB with automatic threshold optimization
clf = TbNB(
    alpha=1.0,
    optimize_threshold=True,
    criterion="balanced_error",
    K=5,
)

clf.fit(X, y)

# Predictions and scores
y_pred = clf.predict(X)
scores = clf.decision_function(X)
```

---

## Threshold optimization

When `optimize_threshold=True`, the decision threshold τ is selected
by cross-validation using the `ThresholdOptimizer`.

The optimizer evaluates a grid of candidate thresholds and selects the
one that optimizes a chosen metric.

Available criteria include:

* `accuracy`
* `precision`
* `recall`
* `specificity`
* `fpr`
* `fnr`
* `f1`
* `mcc`
* `misclassification_error`
* `balanced_error`

The selected threshold is stored in the fitted classifier as `threshold_`.

---

## Iterative refinement (iTb-NB)

If `iterative=True`, Tb-NB applies an iterative refinement procedure
after the global threshold has been selected.

At each iteration:

1. observations close to the current threshold are selected;
2. local class score densities are estimated;
3. a refined threshold is computed from their intersection;
4. the procedure continues until convergence or lack of data.

The result is a sequence of local decision rules that improve classification
in regions of class overlap.

---

## References

Romano, M., Contu, G., Mola, F., & Conversano, C. (2023).
*Threshold-based Naïve Bayes classifier*.
Advances in Data Analysis and Classification, 18, 325–361.
[https://doi.org/10.1007/s11634-023-00536-8](https://doi.org/10.1007/s11634-023-00536-8)

Romano, M., Zammarchi, G., & Conversano, C. (2024).
*Iterative Threshold-Based Naïve Bayes Classifier*.
Statistical Methods & Applications, 33, 235–265.
[https://doi.org/10.1007/s10260-023-00721-1](https://doi.org/10.1007/s10260-023-00721-1)


