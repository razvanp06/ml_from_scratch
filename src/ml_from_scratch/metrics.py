import numpy as np


# linear regression

def mse(y_true, y_pred):
	return ((y_true - y_pred) ** 2).mean()

def rmse(y_true, y_pred):
	return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
	score_model = ((y_pred - y_true) ** 2).sum()
	score_base = ((y_true - y_true.mean()) ** 2).sum()
	return 1 - score_model / score_base


# logistic regression

def confusion_matrix(y_true, y_pred):
	TN = ((y_true == 0) & (y_pred == 0)).sum()
	TP = ((y_true == 1) & (y_pred == 1)).sum()
	FN = ((y_true == 1) & (y_pred == 0)).sum()
	FP = ((y_true == 0) & (y_pred == 1)).sum()

	return np.array([
		[TN, FP],
		[FN, TP]
	])

def accuracy(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)
	score = matrix[0][0] + matrix[1][1]
	return score / len(y_true)

def recall(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)
	if matrix[1].sum() == 0:
		return 0.0

	score = matrix[1][1] / matrix[1].sum()
	return score

def precision(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)
	if matrix[1][1] + matrix[0][1] == 0:
		return 0.0

	score = matrix[1][1] / (matrix[1][1] + matrix[0][1])
	return score

def f1_score(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)
	pre = precision(y_true, y_pred)
	rec = recall(y_true, y_pred)
	if pre + rec == 0:
		return 0.0

	return 2 * pre * rec / (pre + rec)

def roc_curve(y_true, scores):
	indices = np.argsort(-scores)
	sorted_scores = scores[indices]
	sorted_y = y_true[indices]

	TP = np.cumsum(sorted_y)
	tpr = TP / (y_true == 1).sum()
	FP = np.cumsum(1 - sorted_y)
	fpr = FP / (y_true == 0).sum()

	tpr = np.concatenate([[0], tpr])
	fpr = np.concatenate([[0], fpr])
	sorted_scores = np.concatenate([[np.inf], sorted_scores])

	return fpr, tpr, sorted_scores

def roc_auc_score(y_true, scores):
	fpr, tpr, _ = roc_curve(y_true, scores)
	return np.trapezoid(tpr, fpr)

if __name__ == '__main__':
	scores = np.array([0.4, 0.9, 0.1])
	y_true = np.array([1, 1, 0])

	order = np.argsort(-scores)       # [1, 0, 2]: 0.9 is at position 1, 0.4 at 0, 0.1 at 2
	print(scores[order])                     # [0.9, 0.4, 0.1]
	print(y_true[order])                     # [1, 1, 0]