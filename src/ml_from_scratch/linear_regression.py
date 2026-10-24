import numpy as np

class LinearRegression:
	def __init__(self, method='normal', lr=0.01, n_iters=1000):
		self.method = method
		self.lr = lr
		self.n_iters = n_iters
	
	def fit(self, X, y):
		samples, features = X.shape
		X_bias = np.c_[np.ones(samples), X]

		if self.method == 'normal':
			left_side = X_bias.T @ X_bias
			right_side = X_bias.T @ y
			theta = np.linalg.solve(left_side, right_side)
			self.intercept_, self.coef_ = theta[0], theta[1:]
		elif self.method == 'gd':
			self.loss_history_ = []
			theta = np.zeros(X_bias.shape[1])

			for _ in range(self.n_iters):
				preds = X_bias @ theta
				error = preds - y
				self.loss_history_.append((error ** 2).mean())
				gradient = (2 / samples) * X_bias.T @ error
				theta = theta - self.lr * gradient

			self.intercept_, self.coef_ = theta[0], theta[1:]

		return self

if __name__ == '__main__':
	X = np.array([[1], [2], [3]])
	y = np.array([3, 5, 7])
	model = LinearRegression(method='gd').fit(X, y)
	print(model.intercept_, model.coef_, model.loss_history_[0], model.loss_history_[-1])