"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Union
import numpy as np
import pandas as pd
from tree.utils import *

np.random.seed(42)


class TreeNode:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = False
        self.prediction = None
        self.attribute = None
        self.value = None  # threshold for real-valued
        self.children = {}  # for discrete split or {left,right}

    def set_leaf(self, y):
        self.is_leaf = True
        if check_ifreal(y):  # regression
            self.prediction = np.mean(y)
        else:  # classification
            self.prediction = y.mode()[0]


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index", "mse"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not check_ifreal(y):
            if self.criterion == "information_gain":
                self.criterion_used = "entropy"
            elif self.criterion == "gini_index":
                self.criterion_used = "gini_index"
            else:
                raise ValueError("Invalid criterion for classification")
            self.is_regression = False
        else:
            self.criterion_used = "mse"
            self.is_regression = True

        # For discrete input, one-hot encode
        X = one_hot_encoding(X)
        features = list(X.columns)

        def build_tree(X, y, depth):
            node = TreeNode(depth, self.max_depth)

            # stopping conditions
            if depth >= self.max_depth or len(np.unique(y)) == 1 or X.shape[1] == 0:
                node.set_leaf(y)
                return node

            attr, val = opt_split_attribute(X, y, self.criterion_used, features)
            if attr is None:
                node.set_leaf(y)
                return node

            node.attribute = attr
            node.value = val
            splits = split_data(X, y, attr, val)

            for k, (X_sub, y_sub) in splits.items():
                if X_sub.empty:
                    leaf = TreeNode(depth+1, self.max_depth)
                    leaf.set_leaf(y)
                    node.children[k] = leaf
                else:
                    node.children[k] = build_tree(X_sub, y_sub, depth+1)

            return node

        self.root = build_tree(X, y, 0)

    def predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction
        if node.value is None:
            # Discrete attribute
            val = x[node.attribute]
            if val in node.children:
                return self.predict_one(x, node.children[val])
            else:
                return node.prediction
        else:
            # Numeric attribute
            if x[node.attribute] <= node.value:
                return self.predict_one(x, node.children["left"])
            else:
                return self.predict_one(x, node.children["right"])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = one_hot_encoding(X)
        preds = [self.predict_one(X.iloc[i], self.root) for i in range(len(X))]
        return pd.Series(preds)

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.is_leaf:
            print(indent + "Predict:", node.prediction)
            return
        if node.value is None:
            print(indent + f"[{node.attribute}]?")
            for v, child in node.children.items():
                print(indent + f"== {v} =>")
                self.print_tree(child, indent + "   ")
        else:
            print(indent + f"[{node.attribute} <= {node.value}]?")
            print(indent + "Left:")
            self.print_tree(node.children["left"], indent + "   ")
            print(indent + "Right:")
            self.print_tree(node.children["right"], indent + "   ")
        


    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.print_tree()
