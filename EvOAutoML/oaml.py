import random
from copy import deepcopy

import sklearn
from automlstreams.meta import MetaClassifier, LastBestClassifier
from pymfe.mfe import MFE
from scipy import stats
from sklearn.base import ClassifierMixin, MetaEstimatorMixin
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler, train_test_split
from skmultiflow.core import BaseSKMObject
from skmultiflow.trees import HoeffdingTree as HT
from sklearn.naive_bayes import MultinomialNB as MNB
import numpy as np


class EvolutionaryBestClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Classifier that keeps a set of base estimators in a leaderboard
    and pick the estimator for the next window best on the prediction
    accuracy of the estimator in the previous window.

    Parameters
    ----------
    estimators: list of skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator
        default=[DecisionTreeClassifier()]
        A list of estimators for the leaderboard

    window_size: int (default=100)
        The size of the window used for extracting meta-features.

    active_learning: boolean (default=True)
        Switches between using the fit() or partial_fit() method of the base estimators

    Notes
    -----


    """

    def __init__(self,
                 estimator,
                 param_grid,
                 population_size=10,
                 sampling_size=1,
                 window_size=100,
                 metric=accuracy_score,
                 sampling_rate=50,
                 active_learning=True):

        self.estimator = estimator
        self.param_grid = param_grid
        self.population_size = population_size
        self.mutation_size = sampling_size
        self.metric = metric
        self.sampling_rate  = sampling_rate
        #Generate Population
        self.estimators = []
        param_iter = ParameterSampler(self.param_grid, population_size)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]
        for params in param_list:
            new_estimator = sklearn.clone(self.estimator)
            new_estimator.set_params(**params)
            self.estimators.append(new_estimator)

        #self.estimators = self._random_estimators(population_size)
        self.leader_index = 0
        self.window_size = window_size
        self.active_learning = active_learning
        # Actual Index for Dataset
        self.w = 0
        # i is the indicator if already fitted once
        self.i = -1
        #Check if algorithm started
        self._fitted = False

        self.X_window = None
        self.y_window = None

    def _random_estimators(self, n_estimators:int=1):
        param_iter = ParameterSampler(self.param_grid, n_estimators)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]
        estimators = []
        for params in param_list:
            new_estimator = sklearn.clone(self.estimator)
            new_estimator.set_params(**params)
            estimators.append(new_estimator)

        return estimators

    def _mutate_estimator(self,estimator):
        child_estimator = sklearn.clone(estimator)

        key_to_change, value_to_change = random.sample(self.param_grid.items(), 1)[0]
        value_to_change = random.choice(self.param_grid[key_to_change])
        child_estimator.set_params(**{key_to_change: value_to_change})


        return child_estimator

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
            self.X_window = X[-self.window_size:,:]
            self.y_window = y[-self.window_size:]
        else:
            self.X_window = np.roll(self.X_window,-r, axis=0)
            self.y_window = np.roll(self.y_window,-r,axis=0)
            self.X_window[-r:,:] = X
            self.y_window[-r:] = y

            # Train base estimators in a prequential way
        if self.w > 0:
            self.leader_index = self._get_leader_base_estimator_index(X, y)
        try:
            self._partial_fit_estimators(X, y, classes)
        except Exception as e:
            print(e)
        self.w += 1
        self._fitted = True
        self.i = 1
        return self

    def _partial_fit_estimators(self, X, y, classes, sample_weight=None):
        """ Partially (incrementally) fit the base estimators.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model base estimators

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
        # Mutate population
        if self._fitted:
            for m in range(self.mutation_size):
                index = self._get_weakest_base_estimator_index(X,y)
                est = self._mutate_estimator(estimator=self.estimators[index])
                try:
                    est.fit(self.X_window,self.y_window)
                    self.estimators[index] = est
                except:
                    pass

        for index, base_estimator in enumerate(self.estimators):
            try:
                if self.active_learning is True:
                    base_estimator.partial_fit(X, y, classes)
                else:
                    try:
                        base_estimator.fit(X, y, classes)
                    except TypeError as e:
                        base_estimator.fit(X, y)
            except AttributeError as e:
                try:
                    base_estimator.fit(X, y, classes)
                except TypeError as e:
                    base_estimator.fit(X, y)

        return self

    def _get_leader_base_estimator_index(self, X, y):
        """
        Function that returns the index of the best estimator index
        :param X: Features for prediction
        :param y: Ground truth labels
        :return: Integer index of best estimator in self.estimator
        """
        scores = []
        for be in self.estimators:
            try:
                scores.append(self.metric(y,be.predict(X)))
            except:
                scores.append(0.0)

        return scores.index(max(scores))

    def _get_weakest_base_estimator_index(self, X, y):
        """
        Function that returns the index of the least best estimator index
        :param X: Features for prediction
        :param y: Ground truth labels
        :return: Integer index of least best estimator in self.estimator
        """
        scores = []
        for be in self.estimators:
            try:
                scores.append(self.metric(y,be.predict(X)))
            except:
                scores.append(0.0)

        return scores.index(min(scores))

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the X
        entry of the same index. And where the list in index [i] contains len(self.target_values) elements,
        each of which represents the probability that the i-th sample of X belongs to a certain class-label.

        """

        predictions = list()
        for idx,estimator in enumerate(self.estimators):
            try:
                predictions.append(estimator.predict(X))
            except:
                pass
        return stats.mode(predictions)[0][0]

        #return self.estimators[self.leader_index].predict_proba(X)

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """

        predictions = list()
        for idx,estimator in enumerate(self.estimators):
            try:
                predictions.append(estimator.predict(X))
            except:
                pass
        return stats.mode(predictions)[0][0]

        #return self.estimators[self.leader_index].predict(X)

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


class BLASTClassifier(LastBestClassifier):

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

class MetaStreamClassifier(MetaClassifier):

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
