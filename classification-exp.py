import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

plt.show()

X_df = pd.DataFrame(X, columns=["x1", "x2"])
y_s = pd.Series(y)

# Q2 (a)
n = len(X_df)
split_idx = int(0.7 * n)

X_train, X_test = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
y_train, y_test = y_s.iloc[:split_idx], y_s.iloc[split_idx:]

tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Part (a) Results")
print(f"Accuracy: {acc:.3f}")
for cls in np.unique(y_test):
    prec = precision(y_test, y_pred, cls)
    rec = recall(y_test, y_pred, cls)
    print(f"Class {cls}: Precision={prec:.3f}, Recall={rec:.3f}")



# Q2 (b)
max_depth_values = range(1, 11)  # depths
n_outer_splits = 5
n_inner_splits = 5

best_depths = []
outer_scores = []

for outer_iter in range(n_outer_splits):
    # Outer split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, random_state=42 + outer_iter, stratify=y_s
    )

    # Inner Cross Validation
    mean_scores = []

    for depth in max_depth_values:
        inner_fold_scores = []
        for inner_iter in range(n_inner_splits):
            Xi_train, Xi_val, yi_train, yi_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=100 + inner_iter, stratify=y_train
            )

            tree_inner = DecisionTree(criterion="information_gain", max_depth=depth)
            tree_inner.fit(Xi_train, yi_train)
            yi_pred = tree_inner.predict(Xi_val)
            inner_fold_scores.append(accuracy(yi_val, yi_pred))

        mean_scores.append(np.mean(inner_fold_scores))

    best_depth = max_depth_values[int(np.argmax(mean_scores))]
    best_depths.append(best_depth)

    # Retrain with best depth and evaluate on outer test set
    tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    outer_scores.append(accuracy(y_test, y_pred))

print("\n=== Part (b) Nested CV Results ===")
print("Best depths per outer split:", best_depths)
print("Outer split accuracies:", np.round(outer_scores, 3))
print("Mean CV accuracy:", np.mean(outer_scores))
print("Most frequently chosen depth:", pd.Series(best_depths).mode()[0])
