import numpy as np

def sigmoid(z):
	z_clipped = np.clip(z, -500, 500)
	e = np.exp(-z)
	return 1 / (1 + e)


class LogisticRegression:
	def __init__(self, lr=0.1, n_iters=1000, alpha=0.0):
		self.lr = lr
		self.n_iters = n_iters
		self.alpha = alpha

	def fit(self, X, y):
		samples, features = X.shape
		X_bias = np.c_[np.ones(samples), X]
		theta = np.zeros(features + 1)
		self.loss_history_ = []

		for _ in range(self.n_iters):
			theta_reg = theta.copy()
			theta_reg[0] = 0
			z = X_bias @ theta
			p = sigmoid(z)
			p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
			loss = (-y * np.log(p_clipped) - (1 - y) * np.log(1 - p_clipped)).mean()
			self.loss_history_.append(loss)
			gradient = (1 / samples) * (X_bias.T @ (p - y) + self.alpha * theta_reg)
			theta = theta - self.lr * gradient
		
		self.intercept_, self.coef_ = theta[0], theta[1:]
		return self

	def predict_prob(self, X):
		return sigmoid(X @ self.coef_ + self.intercept_)
	
	def predict(self, X, treshold=0.5):
		return (self.predict_prob(X) >= treshold).astype(int)

if __name__ == '__main__':
	from sklearn.linear_model import LogisticRegression as SkLogisticRegression

	for alpha in [0.5, 1, 10]:
		mine = LogisticRegression(lr=0.5, n_iters=20000, alpha=alpha).fit(X_train_s, y_train)
		sk = SkLogisticRegression(C=1 / alpha, max_iter=10000, tol=1e-10).fit(X_train_s, y_train)
		print(alpha, np.abs(mine.coef_ - sk.coef_[0]).max(), abs(mine.intercept_ - sk.intercept_[0]))