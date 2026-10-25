import numpy as np

def mse(y_true, y_pred):
	return ((y_true - y_pred) ** 2).mean()

def rmse(y_true, y_pred):
	return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
	score_model = ((y_pred - y_true) ** 2).sum()
	score_base = ((y_true - y_true.mean()) ** 2).sum()
	return 1 - score_model / score_base

if __name__ == '__main__':
	y_true = np.array([1, 2, 3])
	print(mse(y_true, np.array([1, 2, 4])))
	print(rmse(y_true, np.array([1, 2, 4])))
	print(r2_score(y_true, np.array([1, 2, 4])))
	print(r2_score(y_true, np.array([2, 2, 2])))