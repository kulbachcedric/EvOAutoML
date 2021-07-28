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

class EvolutionaryBestEstimator(base.Estimator):
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
        self.population = deque()
        self.population_metrics = deque()

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

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Estimator:
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self.i % self.sampling_rate == 0:
            idx_best, idx_worst = self._get_best_worst_estimator_index()
            child, child_metric = self._mutate_estimator(estimator=self.population[idx_best])
            del self.population[idx_worst]
            #del self.population_metrics[idx_worst]
            self.population.insert(idx_worst,child)
            #self.population_metrics.insert(idx_worst,child_metric)

        # Update population
        def __update_estimator(idx: int):
            self.population_metrics[idx].update(y_true=y, y_pred=self.population[idx].predict_one(x))
            self.population[idx].learn_one(x=x, y=y, **kwargs)

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

class PipelineHelper(Estimator):

    def __init__(self, models, selected_model=None):
        self.selected_model = None
        self.models = None

        # cloned
        if type(models) == dict:
            self.models = models
        else:
            self.available_models = {}
            for (key, model) in models:
                self.available_models[key] = model

        if selected_model is None:
            self.selected_model = self.available_models[random.choice(list(self.available_models))]
        else:
            self.selected_model = selected_model

    def clone(self):
        return PipelineHelper(self.models, self.selected_model)

    def generate(self, param_dict=None):
        if param_dict is None:
            param_dict = dict()
        per_model_parameters = defaultdict(lambda: defaultdict(list))

        # collect parameters for each specified model
        for k, values in param_dict.items():
            # example:  randomforest__n_estimators
            model_name = k.split('__')[0]
            param_name = k[len(model_name) + 2:]
            if model_name not in self.available_models:
                raise Exception('no such model: {0}'.format(model_name))
            per_model_parameters[model_name][param_name] = values

        ret = []

        # create instance for cartesion product of all available parameters
        # for each model
        for model_name, param_dict in per_model_parameters.items():
            parameter_sets = ParameterGrid(param_dict)
            for parameters in parameter_sets:
                ret.append((model_name, parameters))

        # for every model that has no specified parameters, add default value
        for model_name in self.available_models.keys():
            if model_name not in per_model_parameters:
                ret.append((model_name, dict()))

        return ret

    def _get_params(self, deep=True):
        return self.selected_model._get_params()

    def _set_params(self, new_params: dict = None):
        if len(new_params) > 0:
            self.selected_model = self.available_models[new_params[0]]
            self.selected_model._set_params(new_params=new_params[1])
        elif self.selected_model == None:
            self.selected_model = self.available_models[random.choice(list(self.available_models))]
        return self
