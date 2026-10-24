from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import test_train_split
from ml_from_scratch.linear_regression import LinearRegression

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = test_train_split(X, y, test_size=0.2, random_state=67)

model_normal = LinearRegression(method='normal').fit(X_train, y_train)
model_gd = LinearRegression(method='gd').fit(X_train, y_train)