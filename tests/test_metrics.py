import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as sk_r2_score

from ml_from_scratch.metrics import mse, rmse, r2_score


@pytest.fixture
def predictions():
	rng = np.random.default_rng(0)
	y_true = rng.normal(size=100)
	y_pred = y_true + rng.normal(scale=0.5, size=100)
	return y_true, y_pred


def test_mse_matches_sklearn(predictions):
	y_true, y_pred = predictions
	np.testing.assert_allclose(mse(y_true, y_pred), mean_squared_error(y_true, y_pred))


def test_rmse_matches_sklearn(predictions):
	y_true, y_pred = predictions
	np.testing.assert_allclose(rmse(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)))


def test_r2_matches_sklearn(predictions):
	y_true, y_pred = predictions
	np.testing.assert_allclose(r2_score(y_true, y_pred), sk_r2_score(y_true, y_pred))


def test_hand_computed_values():
	y_true = np.array([1, 2, 3])
	y_pred = np.array([1, 2, 4])
	np.testing.assert_allclose(mse(y_true, y_pred), 1 / 3)
	np.testing.assert_allclose(rmse(y_true, y_pred), np.sqrt(1 / 3))
	np.testing.assert_allclose(r2_score(y_true, y_pred), 0.5)


def test_r2_is_zero_when_predicting_the_mean():
	y_true = np.array([1, 2, 3])
	np.testing.assert_allclose(r2_score(y_true, np.array([2, 2, 2])), 0.0, atol=1e-12)