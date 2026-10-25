import numpy as np
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from ml_from_scratch.preprocessing import StandardScaler


def make_data():
	rng = np.random.default_rng(0)
	return rng.normal(loc=[10, -5, 1000], scale=[2, 0.5, 300], size=(200, 3))


def test_scaled_columns_have_mean_0_std_1():
	X_scaled = StandardScaler().fit_transform(make_data())
	np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
	np.testing.assert_allclose(X_scaled.std(axis=0), 1, atol=1e-10)


def test_matches_sklearn():
	X = make_data()
	mine = StandardScaler().fit(X)
	sk = SkStandardScaler().fit(X)
	np.testing.assert_allclose(mine.mean_, sk.mean_)
	np.testing.assert_allclose(mine.scale_, sk.scale_)
	np.testing.assert_allclose(mine.transform(X), sk.transform(X))


def test_transform_uses_training_statistics():
	X = make_data()
	scaler = StandardScaler().fit(X)
	X_new = X[:10] + 100
	expected = (X_new - X.mean(axis=0)) / X.std(axis=0)
	np.testing.assert_allclose(scaler.transform(X_new), expected)