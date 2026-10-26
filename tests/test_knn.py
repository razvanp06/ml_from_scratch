import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from ml_from_scratch.knn import KNNClassifier
from ml_from_scratch.logistic_regression import LogisticRegression


def plot_border(model, X, y, ax, title):
	x0 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
	x1 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
	g0, g1 = np.meshgrid(x0, x1)
	grid = np.c_[g0.ravel(), g1.ravel()]
	pred = model.predict(grid).reshape(g0.shape)
	ax.contourf(g0, g1, pred, alpha=0.3, cmap='coolwarm')
	ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=12, edgecolors='k', linewidths=0.3)
	ax.set_title(title)


if __name__ == '__main__':
	X_m, y_m = make_moons(n_samples=500, noise=0.3, random_state=42)
	Xm_train, Xm_test, ym_train, ym_test = train_test_split(
		X_m, y_m, test_size=0.2, random_state=42
	)

	# the four models
	models = [
		(KNNClassifier(k=1), "k-NN, k = 1"),
		(KNNClassifier(k=15), "k-NN, k = 15"),
		(KNNClassifier(k=101), "k-NN, k = 101"),
		(LogisticRegression(lr=0.1, n_iters=1000), "Logistic regression"),
	]

	fig, axes = plt.subplots(1, 4, figsize=(18, 4))
	for (model, name), ax in zip(models, axes):
		model.fit(Xm_train, ym_train)
		train_acc = (model.predict(Xm_train) == ym_train).mean()
		test_acc = (model.predict(Xm_test) == ym_test).mean()
		plot_border(model, Xm_train, ym_train, ax, f"{name}\ntrain {train_acc:.2f} / test {test_acc:.2f}")

	plt.tight_layout()
	plt.savefig("knn_borders.png", dpi=120)
	print("saved knn_borders.png")