# ML from Scratch

Classic machine learning algorithms implemented with **NumPy only**, each one checked against scikit-learn on a real dataset.

The goal is to understand what happens inside `.fit()` and `.predict()`: the math, the matrix shapes, and the numerical traps. Each algorithm has a written explanation, an implementation, a comparison with scikit-learn, and a notebook of experiments.

## What's implemented

| Algorithm | File | Highlights |
| --- | --- | --- |
| Linear regression | `linear_regression.py` | normal equation and batch gradient descent, L2 (ridge) regularization |
| Logistic regression | `logistic_regression.py` | sigmoid, log loss, gradient descent, L2 regularization, adjustable decision threshold |
| k-nearest neighbours | `knn.py` | fully vectorized distance matrix (no loops over samples), probability = share of neighbours |
| Feature scaling | `preprocessing.py` | `StandardScaler` with `fit` / `transform` / `fit_transform` |
| Metrics | `metrics.py` | MSE, RMSE, R², confusion matrix, accuracy, precision, recall, F1, ROC curve, ROC-AUC |
| Model selection | `model_selection.py` | k-fold split and cross-validation, works with any model that has `fit` and `predict` |

All models follow the same interface as scikit-learn, so they can be swapped in the same code:

```python
model = LogisticRegression(lr=0.1, n_iters=1000).fit(X_train, y_train)
model.predict_proba(X_test)   # probabilities
model.predict(X_test)         # 0 / 1 labels
model.coef_, model.intercept_ # learned weights (for the regressions)
```

## Project structure

```
ml_from_scratch/
├── src/ml_from_scratch/     # the algorithms (NumPy only)
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── model_selection.py
├── tests/                   # checks against scikit-learn (pytest)
├── notebooks/               # experiments, plots and written conclusions
├── pyproject.toml
└── uv.lock
```

**How the parts work together:**

- **`src/ml_from_scratch/`** contains only the algorithms. They never load data themselves: every model receives `X` and `y` through `fit`, so the same class works on any dataset. Only NumPy is used here.
- **`tests/`** imports the algorithms and compares them with scikit-learn on real data: same weights, same predictions, same metric values, within a tolerance. They also check hand-computed examples.
- **`notebooks/`** imports the same algorithms and runs the experiments: learning-rate curves, regularization paths, threshold choice, decision borders, cross-validation. Each experiment ends with a written conclusion.

## Results

All numbers are on a held-out test set (20% of the data, `random_state=42`).

### Linear regression: California housing

20,640 districts, 8 features, target = median house value (in $100,000).

| Model | Test R² | Test RMSE |
| --- | --- | --- |
| Normal equation | 0.576 | 0.746 (about $75,000) |
| Gradient descent (`lr=0.1`, 1000 steps) | 0.576 | 0.746 |
| scikit-learn `LinearRegression` | 0.576 | 0.746 |

- Both methods reach the same weights as scikit-learn; gradient descent differs only from the 7th decimal on.
- Train R² is 0.613, close to test: the model **underfits** (location and income don't affect price linearly), so ridge regularization barely helps. The best α gives test R² 0.578 at α = 100, then performance drops as the penalty dominates.
- Gradient descent diverges on unscaled features (`Population` reaches the tens of thousands), and converges smoothly after standard scaling.

### Logistic regression: breast cancer

569 tumours, 30 features, 1 = malignant.

| Metric | Test value |
| --- | --- |
| Accuracy | 0.982 |
| Precision | 1.000 (no false alarms) |
| Recall | 0.952 (2 of 42 cancers missed) |
| ROC-AUC | 0.997 |

- The first loss is always ln 2 = 0.6931 (all probabilities start at 0.5), a useful check that the loss is correct.
- With L2 regularization, the weights match scikit-learn's `LogisticRegression(C = 1/alpha)` to within 0.00001.
- Without regularization the weights keep growing with more iterations, because the classes are almost perfectly separable.
- Both mistakes at the default threshold of 0.5 are missed cancers, which is why the threshold should be lowered for screening.

### k-nearest neighbours: breast cancer

| Model | Result |
| --- | --- |
| k-NN, k = 5, test accuracy | 0.956 |
| 5-fold cross-validation, k = 5 | 0.965 (mean) |
| Logistic regression, test accuracy | 0.982 |

- Predictions and probabilities are identical to scikit-learn's `KNeighborsClassifier`.
- Logistic regression wins on this dataset: the classes are separated well by a straight border. k-NN wins on non-linear data such as the two-moons dataset, where it can follow a curved border.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:razvanp06/ml_from_scratch.git
cd ml_from_scratch
uv sync                 # creates .venv and installs everything from uv.lock
uv run pytest -v        # runs the comparisons with scikit-learn
```

To use an algorithm in your own code or notebook:

```python
from ml_from_scratch.preprocessing import StandardScaler
from ml_from_scratch.knn import KNNClassifier
from ml_from_scratch.model_selection import cross_val_score

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit on training data only
X_test_s = scaler.transform(X_test)

scores = cross_val_score(KNNClassifier(k=5), X_train_s, y_train)
```

For notebooks, select the project's `.venv` as the kernel and start with:

```python
%load_ext autoreload
%autoreload 2
```

so edits to `src/` are picked up without restarting the kernel.

## Rules followed in this project

1. **NumPy only** in `src/`. scikit-learn is used only in tests and notebooks, as a reference.
2. **No loops over samples.** All computations over the data are vectorized matrix operations.
3. **Every algorithm is checked** against scikit-learn and against a small example worked out by hand.
4. **The test set is used once**, at the end. Settings like k or α are chosen with cross-validation on the training data.

## Next

- Decision tree (CART) and random forest
- k-means clustering
- Principal component analysis (PCA)
