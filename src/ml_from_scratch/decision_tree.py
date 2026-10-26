import numpy as np

def gini(y):
	if len(y) == 0:
		return 0.0
	p = y.mean()
	return 1 - p ** 2 - (1 - p) ** 2

def best_split(X, y):
	n_samples, n_features = X.shape
	parent = gini(y)
	best_gain, best_feature, best_threshold = 0, None, None

	for j in range(n_features):
		values = np.unique(X[:, j])
		if len(values) < 2:
			continue
		thresholds = (values[:-1] + values[1:]) / 2
		
		for t in thresholds:
			left = X[:, j] <= t
			y_left, y_right = y[left], y[~left]

			weighted = (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / n_samples
			gain = parent - weighted

			if gain > best_gain:
				best_feature, best_threshold, best_gain = j, t, gain

	return best_feature, best_threshold, best_gain


class DecisionTreeClassifier:
	def __init__(self, max_depth=None, min_samples_split=2):
		self.max_depth = max_depth
		self.min_samples_split = 2

	def fit(self, X, y):
		self.root_ = self._grow(X, y, depth=0)
		return self

	def _grow(self, X, y, depth):
		leaf = {'leaf': True, 'value': y.mean(), 'n': len(y)}

		if len(np.unique(y)) == 1:
			return leaf
		if self.max_depth is not None and self.max_depth <= depth:
			return leaf
		if len(y) < self.min_samples_split:
			return leaf

		feature, threshold, gain = best_split(X, y)
		if feature is None:
			return leaf

		mask = X[:, feature] <= threshold
		return {
			'feature': feature,
			'threshold': threshold,
			'left': self._grow(X[mask], y[mask], depth + 1),
			'right': self._grow(X[~mask], y[~mask], depth + 1),
		}

	def _predict_one(self, x):
		node = self.root_
		while 'leaf' not in node:
			if x[node['feature']] <= node['threshold']:
				node = node['left']
			else:
				node = node['right']
		return node['value']

	def predict_proba(self, X):
		return np.array([self._predict_one(x) for x in X])

	def predict(self, X, threshold=0.5):
		return (self.predict_proba(X) >= threshold).astype(int)

	def print_tree(self, feature_names=None):
		self._print_node(self.root_, feature_names, depth=0)

	def _print_node(self, node, feature_names, depth):
		indent = "    " * depth
		if "leaf" in node:
			print(f"{indent}leaf: p = {node['value']:.3f} (n = {node['n']})")
			return
		f = node["feature"]
		name = feature_names[f] if feature_names is not None else f"feature {f}"
		print(f"{indent}{name} <= {node['threshold']:.4g}")
		self._print_node(node["left"], feature_names, depth + 1)
		self._print_node(node["right"], feature_names, depth + 1)

if __name__ == '__main__':
	X = np.array([[1, 5], [2, 1], [3, 4], [4, 2], [5, 6], [6, 3]])
	y = np.array([0, 0, 0, 1, 1, 1])
	print(best_split(X, y))
