"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import numpy as np
import pandas as pd


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding for categorical (discrete) features.
    """
    return pd.get_dummies(X, drop_first=False)


def check_ifreal(y: pd.Series) -> bool:
    """
    Check if the given series has real-valued (continuous) output.
    """
    return (not y.dtype == "category") #or len(y.unique()) >= len(y) // 2  


def entropy(Y: pd.Series) -> float:
    """
    Entropy of class distribution.
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def gini_index(Y: pd.Series) -> float:
    """
    Gini index of class distribution.
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


def mse(Y: pd.Series) -> float:
    """
    Mean squared error (for regression impurity).
    """
    if len(Y) == 0:
        return 0
    mean_val = np.mean(Y)
    return np.mean((Y - mean_val) ** 2)


def information_gain(Y: pd.Series, X_col: pd.Series, criterion: str) -> tuple:
    """
    Calculate best split information gain.
    For discrete: returns (gain, None).
    For real-valued: returns (best_gain, best_threshold).
    criterion: "entropy" | "gini_index" | "mse"
    """
    n = len(Y)

    # Impurity before split
    if criterion == "entropy":
        impurity_before = entropy(Y)
    elif criterion == "gini_index":
        impurity_before = gini_index(Y)
    elif criterion == "mse":
        impurity_before = mse(Y)


    # Case 1: Real-valued feature
    if check_ifreal(X_col):
        order = np.argsort(X_col.to_numpy()) # Gives the indices for the values in the sorted order
        X_sorted = X_col.iloc[order].to_numpy()
        y_sorted = Y.iloc[order].to_numpy()

        unique_vals = np.unique(X_sorted)
        if len(unique_vals) == 1:
            return -np.inf, None
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

        best_gain, best_thresh = -np.inf, None

        # Classification
        if criterion in ["entropy", "gini_index"]:
            classes, y_enc = np.unique(y_sorted, return_inverse=True)
            K = len(classes)

            prefix_counts = np.zeros((len(y_sorted), K), dtype=int)
            for i, c in enumerate(y_enc):
                prefix_counts[i] = prefix_counts[i-1] if i > 0 else 0
                prefix_counts[i, c] += 1
            total_counts = prefix_counts[-1].copy()

            for t in thresholds:
                idx = np.searchsorted(X_sorted, t, side="right")
                if idx == 0 or idx == n:
                    continue

                left_counts = prefix_counts[idx-1]
                right_counts = total_counts - left_counts

                left_size, right_size = idx, n - idx
                if left_size == 0 or right_size == 0:
                    continue

                if criterion == "entropy":
                    left_imp = -np.sum(
                        (left_counts/left_size) * np.log2(left_counts/left_size + 1e-9)
                    )
                    right_imp = -np.sum(
                        (right_counts/right_size) * np.log2(right_counts/right_size + 1e-9)
                    )
                else:  # gini
                    left_imp = 1 - np.sum((left_counts/left_size)**2)
                    right_imp = 1 - np.sum((right_counts/right_size)**2)

                impurity_after = (left_size/n)*left_imp + (right_size/n)*right_imp
                gain = impurity_before - impurity_after

                if gain > best_gain:
                    best_gain, best_thresh = gain, t

        # Regression
        else:  # mse
            y_cumsum = np.cumsum(y_sorted)
            y2_cumsum = np.cumsum(y_sorted**2)

            total_sum = y_cumsum[-1]
            total_sq_sum = y2_cumsum[-1]

            for t in thresholds:
                idx = np.searchsorted(X_sorted, t, side="right")
                if idx == 0 or idx == n:
                    continue

                left_size = idx
                right_size = n - idx

                left_sum = y_cumsum[idx-1]
                right_sum = total_sum - left_sum

                left_sq_sum = y2_cumsum[idx-1]
                right_sq_sum = total_sq_sum - left_sq_sum

                left_mse = left_sq_sum/left_size - (left_sum/left_size)**2
                right_mse = right_sq_sum/right_size - (right_sum/right_size)**2

                impurity_after = (left_size/n)*left_mse + (right_size/n)*right_mse
                gain = impurity_before - impurity_after

                if gain > best_gain:
                    best_gain, best_thresh = gain, t

        return best_gain, best_thresh

    # Case 2: Discrete feature
    values, counts = np.unique(X_col, return_counts=True)
    impurity_after = 0
    for v, c in zip(values, counts):
        Y_sub = Y[X_col == v]
        if criterion == "entropy":
            impurity_after += (c / n) * entropy(Y_sub)
        elif criterion == "gini_index":
            impurity_after += (c / n) * gini_index(Y_sub)
        elif criterion == "mse":
            impurity_after += (c / n) * mse(Y_sub)

    return impurity_before - impurity_after, None



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: list):
    best_gain, best_attr, best_val = -np.inf, None, None

    for attr in features:
        gain, val = information_gain(y, X[attr], criterion)
        if gain > best_gain:
            best_gain, best_attr, best_val = gain, attr, val

    return best_attr, best_val



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Split dataset based on attribute and threshold (if numeric).
    """
    if value is None:
        # Discrete values
        splits = {}
        for v in np.unique(X[attribute]):
            mask = X[attribute] == v # mask is the indices which contain the values which we are splitting
            splits[v] = (X[mask], y[mask])
        return splits
    else:
        # Continuous values
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
        return {
            "left": (X[left_mask], y[left_mask]),
            "right": (X[right_mask], y[right_mask]),
        }
