import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn


# Data cleaning
# Drop irrelevant column
data = data.drop(columns=["car name"])

# Replace '?' in horsepower with NaN and drop rows with missing values
data["horsepower"].replace("?", np.nan, inplace=True)
data = data.dropna()
data["horsepower"] = data["horsepower"].astype(float)

# Features & target
X = data.drop(columns=["mpg"])
y = data["mpg"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# (a) Usage of our DecisionTree
tree = DecisionTree(criterion="mse", max_depth=5)
t0 = time.time()
tree.fit(X_train, y_train)
t1 = time.time()
y_pred = tree.predict(X_test)
t2 = time.time()

rmse = rmse(y_test, y_pred)
mae = mae(y_test, y_pred)

print("Our DecisionTree")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"Fit time for our decision tree: {t1 - t0}")
print(f"Predict time for our decision tree: {t2 - t1}")

# (b) Compare with sklearn DecisionTree
sk_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42)
t0 = time.time()
sk_tree.fit(X_train, y_train)
t1 = time.time()
y_pred_sklearn = sk_tree.predict(X_test)
t2 = time.time()

rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)

print("\nSklearn DecisionTree")
print(f"RMSE: {rmse_sklearn:.3f}")
print(f"MAE: {mae_sklearn:.3f}")
print(f"Fit time for sklearn decision tree: {t1 - t0}")
print(f"Predict time for sklearn decision tree: {t2 - t1}")

# Quick comparison plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, label="Custom Tree", alpha=0.7)
plt.scatter(y_test, y_pred_sklearn, label="Sklearn Tree", alpha=0.7, marker="x")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("True MPG")
plt.ylabel("Predicted MPG")
plt.legend()
plt.title("Custom vs Sklearn Decision Tree Predictions")
plt.show()
