import random
from multiprocessing.pool import ThreadPool

import numpy as np
from river import base
from river import metrics

from EvOAutoML.classification import EvolutionaryBaggingClassifier


class ThreadEvolutionaryBestClassifier(EvolutionaryBaggingClassifier):

    def __init__(self,
                 estimator: base.Estimator,
                 param_grid,
                 population_size=10,
                 sampling_size=1,
                 metric=metrics.Accuracy,
                 sampling_rate=200,
                 seed=42
                 ):
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
        self._initialize_population()



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

        self.pool = ThreadPool()
        self.pool.map(__update_estimator, range(self.population_size))
        self.pool.close()
        #for idx in range(self.population_size):
        #    __update_estimator(idx)

        self.i += 1
        return self


