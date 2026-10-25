import numpy as np

class LinearRegression:
	def __init__(self, method='normal', lr=0.01, n_iters=1000, alpha=0.0):
		self.method = method
		self.lr = lr
		self.n_iters = n_iters
		self.alpha = alpha
	
	def fit(self, X, y):
		samples, features = X.shape
		X_bias = np.c_[np.ones(samples), X]

		if self.method == 'normal':
			eye_matrix = np.eye(features + 1)
			eye_matrix[0, 0] = 0
			left_side = X_bias.T @ X_bias + self.alpha * eye_matrix
			right_side = X_bias.T @ y
			theta = np.linalg.solve(left_side, right_side)
			self.intercept_, self.coef_ = theta[0], theta[1:]
		elif self.method == 'gd':
			self.loss_history_ = []
			theta = np.zeros(X_bias.shape[1])

			for _ in range(self.n_iters):
				theta_ridge = theta.copy()
				theta_ridge[0] = 0
				preds = X_bias @ theta
				error = preds - y
				self.loss_history_.append((error ** 2).mean())
				gradient = (2 / samples) * (X_bias.T @ error + self.alpha * theta_ridge)
				theta = theta - self.lr * gradient

			self.intercept_, self.coef_ = theta[0], theta[1:]

		return self

	def predict(self, X):
		y_hat = X @ self.coef_ + self.intercept_
		return y_hat

if __name__ == '__main__':
	X = np.array([[3], [6], [9]])
	y = np.array([1, 2, 3])
	model = LinearRegression(method='normal').fit(X, y)
	print(model.predict([[12], [30]]))