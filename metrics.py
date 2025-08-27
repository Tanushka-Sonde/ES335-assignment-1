from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size
    assert y.size > 0
    correct = 0
    for i in range(y.size):
        if y.iloc[i] == y_hat.iloc[i]:
            correct += 1

    return correct / y.size * 100


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert y.size > 0

    n = y.size
    tp = 0
    fp = 0
    for i in range(n):
        if y.iloc[i] == cls and y_hat.iloc[i] == cls:
            tp += 1
        elif y_hat.iloc[i] == cls and y.iloc[i] != cls:
            fp += 1

    if tp == 0 and fp == 0:
        return 0
    return tp / (tp + fp)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert y.size > 0
    n = y.size
    tp = 0
    fn = 0
    for i in range(n):
        if y.iloc[i] == cls and y_hat.iloc[i] == cls:
            tp += 1
        elif y_hat.iloc[i] != cls and y.iloc[i] == cls:
            fn += 1

    if tp == 0 and fn == 0:
        return 0
    return tp / (tp + fn)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    assert y.size > 0
    n = y.size
    rmse = 0
    for i in range(n):
        rmse += (y_hat.iloc[i] - y.iloc[i]) ** 2
    rmse /= n
    rmse = rmse ** 0.5
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    assert y.size > 0
    mae = 0
    n = y.size
    for i in range(n):
        mae += abs(y.iloc[i] - y_hat.iloc[i])
    mae /= n
    return mae
