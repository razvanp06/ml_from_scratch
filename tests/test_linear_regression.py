import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from ml_from_scratch.linear_regression import LinearRegression


@pytest.fixture(scope="module")
def california():
	X, y = fetch_california_housing(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	scaler = SkStandardScaler().fit(X_train)
	return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def test_normal_recovers_exact_line():
	X = np.array([[1], [2], [3]])
	y = np.array([3, 5, 7])
	model = LinearRegression(method='normal').fit(X, y)
	np.testing.assert_allclose(model.intercept_, 1.0, atol=1e-10)
	np.testing.assert_allclose(model.coef_, [2.0], atol=1e-10)


def test_gd_recovers_exact_line():
	X = np.array([[1], [2], [3]])
	y = np.array([3, 5, 7])
	model = LinearRegression(method='gd', lr=0.05, n_iters=5000).fit(X, y)
	np.testing.assert_allclose(model.intercept_, 1.0, atol=1e-6)
	np.testing.assert_allclose(model.coef_, [2.0], atol=1e-6)


def test_normal_matches_sklearn(california):
	X_train, X_test, y_train, y_test = california
	mine = LinearRegression(method='normal').fit(X_train, y_train)
	sk = SkLinearRegression().fit(X_train, y_train)
	np.testing.assert_allclose(mine.intercept_, sk.intercept_, atol=1e-8)
	np.testing.assert_allclose(mine.coef_, sk.coef_, atol=1e-8)


def test_gd_matches_normal(california):
	X_train, X_test, y_train, y_test = california
	gd = LinearRegression(method='gd', lr=0.1, n_iters=5000).fit(X_train, y_train)
	normal = LinearRegression(method='normal').fit(X_train, y_train)
	np.testing.assert_allclose(gd.intercept_, normal.intercept_, atol=1e-4)
	np.testing.assert_allclose(gd.coef_, normal.coef_, atol=1e-4)


def test_gd_loss_decreases(california):
	X_train, X_test, y_train, y_test = california
	model = LinearRegression(method='gd', lr=0.1, n_iters=500).fit(X_train, y_train)
	assert len(model.loss_history_) == 500
	assert model.loss_history_[-1] < model.loss_history_[0]


@pytest.mark.parametrize("alpha", [0.1, 10, 1000])
def test_ridge_normal_matches_sklearn(california, alpha):
	X_train, X_test, y_train, y_test = california
	mine = LinearRegression(method='normal', alpha=alpha).fit(X_train, y_train)
	sk = Ridge(alpha=alpha).fit(X_train, y_train)
	np.testing.assert_allclose(mine.intercept_, sk.intercept_, atol=1e-8)
	np.testing.assert_allclose(mine.coef_, sk.coef_, atol=1e-8)


def test_ridge_gd_matches_normal(california):
	X_train, X_test, y_train, y_test = california
	gd = LinearRegression(method='gd', lr=0.1, n_iters=5000, alpha=10).fit(X_train, y_train)
	normal = LinearRegression(method='normal', alpha=10).fit(X_train, y_train)
	np.testing.assert_allclose(gd.intercept_, normal.intercept_, atol=1e-4)
	np.testing.assert_allclose(gd.coef_, normal.coef_, atol=1e-4)


def test_ridge_zero_alpha_equals_plain(california):
	X_train, X_test, y_train, y_test = california
	plain = LinearRegression(method='normal').fit(X_train, y_train)
	ridge = LinearRegression(method='normal', alpha=0.0).fit(X_train, y_train)
	np.testing.assert_allclose(ridge.coef_, plain.coef_, atol=1e-12)


def test_predict_shape(california):
	X_train, X_test, y_train, y_test = california
	model = LinearRegression(method='normal').fit(X_train, y_train)
	assert model.predict(X_test).shape == (len(X_test),)