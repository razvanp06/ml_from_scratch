import numpy as np

from ml_from_scratch.decision_tree import DecisionTreeClassifier

def bootstrap_sample(n_samples, rng):
	sample_idx = rng.integers(0, n_samples, size=n_samples)
	oob_idx = np.setdiff1d(np.arange(n_samples), sample_idx)
	return sample_idx, oob_idx

class RandomForestClassifier:
	def __init__(self, n_trees=100, max_features="sqrt", max_depth=None, seed=67):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth
		self.seed = seed

	def fit(self, X, y):
		n_samples, n_features = X.shape
		rng = np.random.default_rng(self.seed)

		if self.max_features == "sqrt":
			n_feat = max(1, int(np.sqrt(n_features)))
		elif self.max_features is None:
			n_feat = n_features
		else:
			n_feat = min(self.max_features, n_features)

		self.trees_ = []
		self.oob_indices_ = []

		for _ in range(self.n_trees):
			sample_idx, oob_idx = bootstrap_sample(n_samples, rng)
			tree_seed = rng.integers(1_000_000_000)
			tree = DecisionTreeClassifier(
				max_depth=self.max_depth,
				max_features=n_feat,
				seed=tree_seed,
			)
			tree.fit(X[sample_idx], y[sample_idx])
			self.trees_.append(tree)
			self.oob_indices_.append(oob_idx)

		self._compute_oob_score(X, y)
		return self

	def _compute_oob_score(self, X, y):
		n_samples = X.shape[0]
		oob_sum = np.zeros(n_samples)
		oob_count = np.zeros(n_samples)

		for tree, oob_idx in zip(self.trees_, self.oob_indices_):
			if len(oob_idx) == 0:
				continue
			oob_sum[oob_idx] += tree.predict_proba(X[oob_idx])
			oob_count[oob_idx] += 1

		covered = oob_count > 0
		oob_proba = oob_sum[covered] / oob_count[covered]
		oob_pred = (oob_proba >= 0.5).astype(int)

		self.oob_score_ = (oob_pred == y[covered]).mean()
		self.oob_coverage_ = covered.mean()

	def predict_proba(self, X):
		return np.mean([tree.predict_proba(X) for tree in self.trees_], axis=0)

	def predict(self, X, threshold=0.5):
		return (self.predict_proba(X) >= threshold).astype(int)