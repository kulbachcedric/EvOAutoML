import copy
import random
import typing
from multiprocessing.pool import ThreadPool, Pool

import pandas as pd
from river import metrics, compose
from river.metrics import ClassificationMetric
from river import base
from sklearn.model_selection import ParameterSampler
from collections import deque, defaultdict


class EvolutionaryBestClassifier(base.Classifier):

    def __init__(self,
                 estimator: base.Estimator,
                 param_grid,
                 population_size=10,
                 sampling_size=1,
                 metric=metrics.Accuracy,
                 sampling_rate=200,
                 ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.population_size = population_size
        self.sampling_size = sampling_size
        self.metric = metric
        self.sampling_rate = sampling_rate
        self.n_jobs = 5

        self.i = 0
        self.population = deque()
        self.population_metrics = deque()

        self.initialize_population()



    def initialize_population(self):
        """

        :return:
        """
        # Generate Population
        self.population = []
        param_iter = ParameterSampler(self.param_grid, self.population_size)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]

        nested_params = defaultdict(dict)
        for params in param_list:
            for key, value in params.items():
                key, delim, sub_key = key.partition('__')
                if delim:
                    nested_params[key][sub_key] = value

            new_estimator = self.estimator.clone()
            new_estimator = new_estimator._set_params(nested_params)
            self.population.append(new_estimator)
            self.population_metrics.append(self.metric())

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        predictions = list()
        for idx, estimator in enumerate(self.population):
            predictions.append(estimator.predict_proba_one(x))
        return dict(pd.DataFrame(predictions).mean())
        # return predictions[0]

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
        pass

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
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
            self.population_metrics[idx].update(y_true=y, y_pred=self.population[idx].predict_one(x))
            self.population[idx].learn_one(x=x, y=y)

        #self.pool = ThreadPool()
        #results = self.pool.map(__update_estimator, list(range(self.population_size)),chunksize=10)
        #self.pool.close()
        #self.pool.join()

        for idx in range(self.population_size):
            __update_estimator(idx)

        self.i += 1
        return self

    def get_estimator_with_parameters(self, param_dict):
        estimator = self.estimator.clone()
        for param in param_dict.keys():
            estimator_key, parameter_key = param.split('__')
            setattr(estimator.steps[estimator_key], parameter_key, param_dict[param])
        return estimator

    def clone(self):
        return copy.deepcopy(self)

    def _mutate_estimator(self, estimator) -> (base.Classifier, ClassificationMetric):
        child_estimator = estimator.clone()
        key_to_change, value_to_change = random.sample(self.param_grid.items(), 1)[0]
        value_to_change = random.choice(self.param_grid[key_to_change])
        child_estimator._set_params({key_to_change: value_to_change})
        # todo refactor Mutation
        return child_estimator, self.metric()

    def _get_best_worst_estimator_index(self):
        scores = []
        for be in self.population_metrics:
            scores.append(be.get())
        return scores.index(max(scores)), scores.index(min(scores))

    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        # self.estimators = [be.reset() for be in self.estimators]
        self.i = -1
        return self
