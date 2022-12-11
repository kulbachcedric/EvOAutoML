import copy

import numpy as np
from river import base
from sklearn.model_selection import ParameterSampler


class EvolutionaryBaggingEstimator(base.Wrapper, base.Ensemble):
    def __init__(
        self,
        model,
        param_grid,
        metric,
        population_size=10,
        sampling_size=1,
        sampling_rate=1000,
        seed=42,
    ):
        self._rng = np.random.RandomState(seed)
        param_iter = ParameterSampler(
            param_grid, population_size, random_state=self._rng
        )
        param_list = list(param_iter)
        param_list = [{k: v for (k, v) in d.items()} for d in param_list]
        super().__init__(
            [
                self._initialize_model(model=model, params=params)
                for params in param_list
            ]
        )
        self.param_grid = param_grid
        self.population_size = population_size
        self.sampling_size = sampling_size
        self.metric = metric
        self.sampling_rate = sampling_rate
        self.n_models = population_size
        self.model = model
        self.seed = seed
        self._i = 0
        self._population_metrics = [
            copy.deepcopy(metric()) for _ in range(self.n_models)
        ]

    @property
    def _wrapped_model(self):
        return self.model

    def _initialize_model(self, model: base.Estimator, params):
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
            self._population_metrics[idx_worst] = copy.deepcopy(self.metric())

        for idx, model in enumerate(self):
            self._population_metrics[idx].update(
                y_true=y, y_pred=model.predict_one(x)
            )
            for _ in range(self._rng.poisson(6)):
                model.learn_one(x, y)
        self._i += 1
        return self

    def reset(self):
        """Resets the estimator to its initial state.

        Returns
        -------
            self

        """
        # self.estimators = [be.reset() for be in self.estimators]
        self._i = 0
        return self

    def _mutate_estimator(self, estimator) -> (base.Classifier):
        child_estimator = copy.deepcopy(estimator)
        key_to_change = self._rng.choice(list(self.param_grid.keys()))
        value_to_change = self.param_grid[key_to_change][
            self._rng.choice(range(len(self.param_grid[key_to_change])))
        ]
        child_estimator._set_params({key_to_change: value_to_change})
        return child_estimator

    def clone(self):
        """Return a fresh estimator with the same parameters.

        The clone has the same parameters but has not been
        updated with any data.

        This works by looking at the parameters from the class signature.
        Each parameter is either

        - recursively cloned if it's a River classes.
        - deep-copied via `copy.deepcopy` if not.

        If the calling object is stochastic (i.e. it accepts a seed parameter)
        and has not been seeded, then the clone will not be idempotent.
        Indeed, this method's purpose if simply to return a new instance with
        the same input parameters.

        """
        return copy.deepcopy(self)


class EvolutionaryBaggingOldestEstimator(EvolutionaryBaggingEstimator):
    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self._i % self.sampling_rate == 0:
            scores = [be.get() for be in self._population_metrics]
            idx_best = scores.index(max(scores))
            # idx_worst = scores.index(min(scores))
            child = self._mutate_estimator(estimator=self[idx_best])
            self.models.pop(0)
            self.models.append(child)
            self._population_metrics.pop(0)
            self._population_metrics.append(copy.deepcopy(self.metric()))

        for idx, model in enumerate(self):
            y_pred = model.predict_one(x)
            if y_pred is not None and y_pred != {}:
                self._population_metrics[idx].update(y_true=y, y_pred=y_pred)
            for _ in range(self._rng.poisson(6)):
                model.learn_one(x, y)
        self._i += 1
        return self
