import collections
import copy
from typing import Iterable, Union, Tuple, Any, Type

import pandas as pd
import ray
import river.base
from river import metrics
from river.metrics import ClassificationMetric
from sklearn.model_selection import ParameterSampler
from collections import deque
import numpy as np
import random
import typing
from collections import defaultdict
from river import base
from river.base import Estimator
from sklearn.model_selection import ParameterGrid

from EvOAutoML.classification import EvolutionaryBestClassifier


class DecentralizedEvolutionaryBestClassifier(EvolutionaryBestClassifier):
    def __init__(self,
                 estimator: base.Estimator,
                 param_grid,
                 population_size=10,
                 sampling_size=1,
                 metric=metrics.Accuracy,
                 sampling_rate=200,
                 seed=42
                 ):
        ray.init()
        self.estimator = estimator
        self.param_grid = param_grid
        self.population_size = population_size
        self.sampling_size = sampling_size
        self.metric = metric
        self.sampling_rate = sampling_rate

        self.i = 0
        random.seed(seed)
        np.random.seed(seed)
        self.population = []
        self.population_metrics = []

        self.__initialize_population()

    def __initialize_population(self):
        """

        :return:
        """
        # Generate Population
        self.population = []
        param_iter = ParameterSampler(self.param_grid, self.population_size)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]
        for params in param_list:
            new_estimator = self.estimator.clone()
            new_estimator = new_estimator._set_params(params)
            self.population.append(RayClassifier.remote(new_estimator))
            self.population_metrics.append(self.metric())

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        """Averages the predictions of each classifier."""
        y_pred = collections.Counter()
        preds = ray.get([ind.predict_proba_one.remote(x) for ind in self.population])
        for p in preds:
            y_pred.update(p)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Estimator:
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self.i % self.sampling_rate == 0:
            idx_best, idx_worst = self._get_best_worst_estimator_index()
            child, child_metric = self._mutate_estimator(estimator=ray.get(self.population[idx_best].get_classifier.remote()))
            del self.population[idx_worst]
            del self.population_metrics[idx_worst]
            self.population.insert(idx_worst,RayClassifier.remote(child))
            self.population_metrics.insert(idx_worst,child_metric)

        # Update population
        def __update_estimator(idx: int):
            self.population_metrics[idx].update(y_true=y, y_pred=ray.get(self.population[idx].predict_one.remote(x)))
            self.population[idx].learn_one.remote(x=x, y=y, **kwargs)

        for idx in range(self.population_size):
            __update_estimator(idx)

        self.i += 1
        return self

    def _get_best_worst_estimator_index(self):
        scores = [be.get() for be in self.population_metrics]
        return scores.index(max(scores)), scores.index(min(scores))

    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        # self.estimators = [be.reset() for be in self.estimators]
        self.i = 0
        self.__initialize_population()
        return self

@ray.remote
class RayClassifier(river.base.Classifier):

    def __init__(self, classifier:river.base.Classifier):
        self.classifier = classifier

    def get_classifier(self):
        return self.classifier

    def set_classifier(self, classifier:river.base.Classifier):
        self.classifier = classifier
        return self

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
        self.classifier.learn_one(x,y,**kwargs)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        return self.classifier.predict_proba_one(x)


    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        return super().predict_one(x)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        self.classifier.predict_proba_many(X)

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        return super().predict_many(X)
