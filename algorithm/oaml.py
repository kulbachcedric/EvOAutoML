import copy
import random
import typing
from copy import deepcopy

import pandas as pd
from river import metrics, compose
from river.metrics import ClassificationMetric
from river.utils import dict2numpy
from scipy import stats
import numpy as np
from river import base
from sklearn.model_selection import ParameterSampler
from collections import deque


class EvolutionaryBestClassifier(base.Classifier):
    """ Classifier that keeps a set of base estimators in a leaderboard
    and pick the estimator for the next window best on the prediction
    accuracy of the estimator in the previous window.

    Parameters
    ----------


    window_size: int (default=100)
        The size of the window used for extracting meta-features.


    Notes
    -----


    """

    def __init__(self,
             estimator: base.Estimator,
             param_grid,
             population_size=10,
             sampling_size=1,
             window_size=100,
             metric=metrics.Accuracy,
             sampling_rate=50,
            ):

        self.estimator = estimator
        self.param_grid = param_grid
        self.population_size = population_size
        self.sampling_size = sampling_size
        self.metric = metric
        self.sampling_rate  = sampling_rate
        self.window_size = window_size

        self.i = 0
        self.X_window = deque()
        self.y_window = deque()
        self.population = deque()
        self.population_metrics = deque()

        self.initialize_population()

    def initialize_population(self):
        """

        :return:
        """
        #Generate Population
        self.population = []
        param_iter = ParameterSampler(self.param_grid, self.population_size)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]
        for params in param_list:
            new_estimator = self.estimator._set_params(params)
            self.population.append(new_estimator)
            self.population_metrics.append(self.metric())

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        predictions = list()
        for idx,estimator in enumerate(self.population):
            predictions.append(estimator.predict_proba_one(x))
        return dict(pd.DataFrame(predictions).mean())

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
        pass

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self.i >= self.window_size:
            if self.i % self.sampling_rate == 0:
                idx_best = self._get_leader_base_estimator_index()
                idx_worst = self._get_weakest_base_estimator_index()
                child, child_metric = self._mutate_estimator(estimator=self.population[idx_best])
                for idx, x_item_window in enumerate(self.X_window):
                    y_item_window = self.y_window[idx]
                    child_metric.update(y_true=y_item_window, y_pred=child.predict_one(x_item_window))
                    child.learn_one(x=x_item_window,y=y_item_window)
                del self.population[idx_worst]
                del self.population_metrics[idx_worst]
                self.population.append(child)
                self.population_metrics.append(child_metric)
        # Update Population
        for idx, estimator in enumerate(self.population):
            self.population_metrics[idx].update(y_true=y, y_pred=estimator.predict_one(x))
            estimator.learn_one(x=x, y=y)

        self.X_window.append(x)
        self.y_window.append(y)
        if self.i >= self.window_size:
            self.X_window.popleft()
            self.y_window.popleft()

        self.i += 1
        return self


    def get_estimator_with_parameters(self, param_dict):
        estimator = self.estimator.clone()
        for param in param_dict.keys():
            estimator_key, parameter_key = param.split('__')
            setattr(estimator.steps[estimator_key],parameter_key,param_dict[param])
        return estimator

    def clone(self):
        return copy.deepcopy(self)

    def _mutate_estimator(self,estimator) -> (base.Classifier, ClassificationMetric):
        child_estimator = estimator.clone()
        key_to_change, value_to_change = random.sample(self.param_grid.items(), 1)[0]
        value_to_change = random.choice(self.param_grid[key_to_change])
        child_estimator._set_params({key_to_change: value_to_change})
        return child_estimator, self.metric()

    def _get_population_scores(self):
        scores = []
        for be in self.population_metrics:
            scores.append(be.get())
        return scores

    def _get_leader_base_estimator_index(self):
        """
        Function that returns the index of the best estimator index
        :param X: Features for prediction
        :param y: Ground truth labels
        :return: Integer index of best estimator in self.estimator
        """
        scores = self._get_population_scores()
        return scores.index(max(scores))

    def _get_weakest_base_estimator_index(self):
        """
        Function that returns the index of the least best estimator index
        :param X: Features for prediction
        :param y: Ground truth labels
        :return: Integer index of least best estimator in self.estimator
        """
        scores = self._get_population_scores()
        return scores.index(min(scores))

    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        #self.estimators = [be.reset() for be in self.estimators]
        self.leader_index = 0
        self.w = 0
        self.i = -1
        self.X_window = None
        self.y_window = None
        self._fitted = False
        return self


class BLASTClassifier(base.Classifier):

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. Usage varies depending on the learning method.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
            self

        """
        r, c = X.shape
        # Create Dataset if not initialized
        if self.i < 0:
            self.X_window = np.zeros((self.window_size, c))
            self.y_window = np.zeros(self.window_size)
        # Overwrite window with new Data
        # Check if incomming data is bigger than shape
        if r > self.window_size:
            self.X_window = X[-self.window_size:, :]
            self.y_window = y[-self.window_size:]
        else:
            self.X_window = np.roll(self.X_window, -r, axis=0)
            self.y_window = np.roll(self.y_window, -r, axis=0)
            self.X_window[-r:, :] = X
            self.y_window[-r:] = y

        # Train base estimators in a prequential way
        if self.w > 0:
            self.leader_index = self._get_leader_base_estimator_index(self.X_window, self.y_window)

        self._partial_fit_estimators(X, y, classes)
        self.w += 1
        self.i = 1
        return self

class MetaStreamClassifier(base.Classifier):

    def partial_fit(self, X, y=None, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. Usage varies depending on the learning method.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.

        Returns
        -------
            self

        """
        r, c = X.shape
        # Create Dataset if not initialized
        if self.i < 0:
            self.X_window = np.zeros((self.window_size, c))
            self.y_window = np.zeros(self.window_size)
        # Overwrite window with new Data
        # Check if incomming data is bigger than shape
        if r > self.window_size:
            self.X_window = X[-self.window_size:, :]
            self.y_window = y[-self.window_size:]
        else:
            self.X_window = np.roll(self.X_window, -r, axis=0)
            self.y_window = np.roll(self.y_window, -r, axis=0)
            self.X_window[-r:, :] = X
            self.y_window[-r:] = y

        # Extract meta-features
        mfe = MFE(self.mfe_groups, suppress_warnings=True).fit(self.X_window, self.y_window)
        metafeatures = np.array([mfe.extract()[1]])
        metafeatures[~np.isfinite(metafeatures)] = 0


        # Select leader for predictions
        if self.w > 0:
            predicted = self.meta_estimator.predict(metafeatures)
            self.leader_index = predicted[0]

        # Train base estimators
        X_window_train, X_window_test, y_window_train, y_window_test = train_test_split(X, y)
        self._partial_fit_base_estimators(X_window_train, y_window_train, classes)
        leader_index = self._get_leader_base_estimator_index(X_window_test, y_window_test)

        # Train meta learner
        metaclasses = [c for c in range(len(self.base_estimators))]
        self.meta_estimator.partial_fit(metafeatures, [leader_index], metaclasses)
        self.i = 1
        self.w += 1

        return self
