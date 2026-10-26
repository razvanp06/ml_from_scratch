import numpy as np

def pairwise_distances(A, B):
	A_sq = (A ** 2).sum(axis=1)[:, None]
	B_sq = (B ** 2).sum(axis=1)[None, :]
	dots = A @ B.T
	dots_sq = A_sq + B_sq - 2 * dots
	return np.sqrt(np.maximum(dots_sq, 0))


class KNNClassifier:
	def __init__(self, k=5):
		self.k = k

	def fit(self, X, y):
		self.X_train_ = X
		self.y_train_ = y
		return self

	def predict_prob(self, X):
		D = pairwise_distances(X, self.X_train_)
		cols = np.argsort(D, axis=1)[:, :self.k]
		labels = self.y_train_[cols]
		return labels.mean(axis=1)

	def predict(self, X, threshold=0.5):
		return (self.predict_prob(X) >= threshold).astype(int)

if __name__ == '__main__':
	A = np.array([[0, 0], [1, 1]])
	B = np.array([[3, 4], [1, 1], [0, 1]])
	print(pairwise_distances(A, B))