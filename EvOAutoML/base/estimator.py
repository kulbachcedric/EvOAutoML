import copy
import random
from collections import defaultdict

import numpy as np
from river import base
from river import metrics
from river.base import Estimator
from river.metrics import ClassificationMetric
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from river import ensemble

class Individual():

    def __init__(self, estimator:base.Estimator):
        pass


class EvolutionaryBaggingEstimator(base.WrapperMixin, base.EnsembleMixin):

    def __init__(self, model,
                 param_grid,
                 population_size=10,
                 sampling_size=1,
                 metric=metrics.Accuracy,
                 sampling_rate=1000,
                 seed=42):

        param_iter = ParameterSampler(param_grid, population_size)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]
        super().__init__(self._initialize_model(model=model,params=params) for params in param_list)
        self.param_grid = param_grid
        self.population_size = population_size
        self.sampling_size = sampling_size
        self.metric = metric
        self.sampling_rate = sampling_rate
        self.n_models = population_size
        self.model = model
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._i = 0
        #self._drift_detectors = [copy.deepcopy(ADWIN()) for _ in range(self.n_models)]
        self._population_metrics = [copy.deepcopy(metric()) for _ in range(self.n_models)]


    @property
    def _wrapped_model(self):
        return self.model


    def _initialize_model(self,model:base.Estimator,params):
        model = copy.deepcopy(model)
        model._set_params(params)
        return model

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self._i % self.sampling_rate == 0:
            scores = [be.get() for be in self._population_metrics]
            idx_best = scores.index(max(scores))
            idx_worst = scores.index(min(scores))
            child = self._mutate_estimator(estimator=self[idx_best])
            self.models[idx_worst] = child
            #self.population_metrics[idx_worst] = copy.deepcopy(self.metric())

        for idx, model in enumerate(self):
            self._population_metrics[idx].update(y_true=y, y_pred=model.predict_one(x))
            for _ in range(self._rng.poisson(5)):
                model.learn_one(x, y)
        self._i += 1
        return self


    def reset(self):
        """ Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        # self.estimators = [be.reset() for be in self.estimators]
        self._i = 0
        return self

    def _mutate_estimator(self, estimator) -> (base.Classifier, ClassificationMetric):
        child_estimator = copy.deepcopy(estimator)
        key_to_change, value_to_change = random.sample(self.param_grid.items(), 1)[0]
        value_to_change = random.choice(self.param_grid[key_to_change])
        child_estimator._set_params({key_to_change: value_to_change})
        # todo refactor Mutation
        return child_estimator

    def clone(self):
        """Return a fresh estimator with the same parameters.

        The clone has the same parameters but has not been updated with any data.

        This works by looking at the parameters from the class signature. Each parameter is either

        - recursively cloned if it's a River classes.
        - deep-copied via `copy.deepcopy` if not.

        If the calling object is stochastic (i.e. it accepts a seed parameter) and has not been
        seeded, then the clone will not be idempotent. Indeed, this method's purpose if simply to
        return a new instance with the same input parameters.

        """
        return copy.deepcopy(self)

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
        return PipelineHelper(self.models)#, self.selected_model.clone())

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
            self.selected_model = self.available_models[new_params[0]].__class__(**new_params[1])
        elif self.selected_model == None:
            self.selected_model = self.available_models[random.choice(list(self.available_models))]
        return self
