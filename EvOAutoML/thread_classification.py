import collections
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool

from EvOAutoML.base.estimator import EvolutionaryBestEstimator

import river
import copy
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

pool = Pool(5)

class ThreadEvolutionaryBestClassifier(EvolutionaryBestClassifier):

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Estimator:
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self.i % self.sampling_rate == 0:
            idx_best, idx_worst = self._get_best_worst_estimator_index()
            child, child_metric = self._mutate_estimator(estimator=self.population[idx_best])
            del self.population[idx_worst]
            del self.population_metrics[idx_worst]
            self.population.insert(idx_worst,child)
            self.population_metrics.insert(idx_worst,child_metric)

        # Update population
        def __update_estimator(idx: int):
            y_pred = self.population[idx].predict_one(x)
            if y_pred != {} and y_pred is not None:
                self.population_metrics[idx].update(y_true=y, y_pred=y_pred)
            self.population[idx].learn_one(x=x, y=y, **kwargs)

        for idx in range(self.population_size):
            __update_estimator(idx)

        self.i += 1
        return self