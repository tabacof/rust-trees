from typing import List

import numpy as np
from rustrees.rustrees import DecisionTree as rt_dt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .utils import prepare_dataset


class DecisionTree(BaseEstimator):
    """
    A decision tree model implemented using Rust.
    Options for regression and classification are available.
    """

    def __init__(
        self,
        min_samples_leaf=1,
        max_depth: int = 10,
        max_features: int = None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node. The default is 1.
        max_depth : int, optional
            The maximum depth of the tree. The default is 10.
        max_features: int, optional
            The maximum number of features per split. Default is None, which means all features are considered.
        random_state : int, optional
            The seed used by the random number generator. The default is None.
        """
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : pd.DataFrame or 2D array-like object
            The features.
        y : list, Numpy array, or Pandas Series
            The target.
        """
        raise NotImplementedError()

    def predict(self, X) -> List:
        """
        Predict values (regression) or class (classification) for X.

        Parameters
        ----------
        X : pd.DataFrame or 2D array-like object
            The features.

        Returns
        -------
        List
            The predicted values or classes.
        """
        raise NotImplementedError()

    def predict_proba(self, X) -> List:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : pd.DataFrame or 2D array-like object
            The features.

        Returns
        -------
        List
            The predicted class probabilities.
        """
        raise NotImplementedError()


class DecisionTreeRegressor(DecisionTree, ClassifierMixin):
    def __init__(self, **kwargs):
        super(DecisionTreeRegressor, self).__init__(**kwargs)

    def fit(self, X, y) -> "DecisionTreeRegressor":
        dataset = prepare_dataset(X, y)
        self.tree = rt_dt.train_reg(
            dataset,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        return self

    def predict(self, X) -> List:
        dataset = prepare_dataset(X)
        return self.tree.predict(dataset)


class DecisionTreeClassifier(DecisionTree, RegressorMixin):
    def __init__(self, **kwargs):
        super(DecisionTreeClassifier, self).__init__(**kwargs)

    def fit(self, X, y) -> "DecisionTreeClassifier":
        dataset = prepare_dataset(X, y)
        self.tree = rt_dt.train_clf(
            dataset,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        return self

    def predict(self, X, threshold: float = 0.5) -> List:
        dataset = prepare_dataset(X)
        return (np.array(self.tree.predict(dataset)) > threshold).astype(int)

    def predict_proba(self, X) -> List:
        dataset = prepare_dataset(X)
        return self.tree.predict(dataset)
