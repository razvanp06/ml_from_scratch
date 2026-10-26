from ml_from_scratch.knn import KNNClassifier
import numpy as np

def k_fold_indices(n_samples, n_folds=5, seed=67):
	rng = np.random.default_rng(seed)
	shuffled = rng.permutation(n_samples)
	pieces = np.array_split(shuffled, n_folds)

	folds = []
	for i in range(n_folds):
		valid_idx = pieces[i]
		train_idx = np.concatenate([pieces[j] for j in range(n_folds) if j != i])
		folds.append((train_idx, valid_idx))

	return folds

def cross_val_score(model, X, y, n_folds=5, seed=67):
	folds = k_fold_indices(X.shape[0], n_folds, seed)
	scores = []

	for (train_idx, valid_idx) in folds:
		model.fit(X[train_idx], y[train_idx])
		y_pred = model.predict(X[valid_idx])
		scores.append((y_pred == y[valid_idx]).mean())

	return scores

