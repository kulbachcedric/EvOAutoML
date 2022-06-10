import copy
import random
from collections import defaultdict

import numpy as np
from river import base, compose, preprocessing, tree
from river import metrics
from river.base import Estimator
from river.drift import ADWIN
from river.metrics import ClassificationMetric
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler



class EvolutionaryBaggingEstimator(base.Wrapper, base.Ensemble):

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
        super().__init__([self._initialize_model(model=model,params=params) for params in param_list])
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

    @classmethod
    def _unit_test_params(cls):
        model = tree.HoeffdingTreeClassifier()

        param_grid = {
            'max_depth': [10, 30, 60, 10, 30, 60],
            },

        yield {
            "model": model,
            "param_grid": param_grid,
        }

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
            self._population_metrics[idx_worst] = copy.deepcopy(self.metric())

        for idx, model in enumerate(self):
            self._population_metrics[idx].update(y_true=y, y_pred=model.predict_one(x))
            for _ in range(self._rng.poisson(6)):
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


class EvolutionaryBaggingOldestEstimator(EvolutionaryBaggingEstimator):

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self._i % self.sampling_rate == 0:
            scores = [be.get() for be in self._population_metrics]
            idx_best = scores.index(max(scores))
            idx_worst = scores.index(min(scores))
            child = self._mutate_estimator(estimator=self[idx_best])
            self.models.pop(0)
            self.models.append(child)
            self._population_metrics.pop(0)
            self._population_metrics.append(copy.deepcopy(self.metric()))

        for idx, model in enumerate(self):
            self._population_metrics[idx].update(y_true=y, y_pred=model.predict_one(x))
            for _ in range(self._rng.poisson(6)):
                model.learn_one(x, y)
        self._i += 1
        return self

class EvolutionaryLeveragingBaggingEstimator(base.Wrapper, base.Ensemble):
    """Leveraging Bagging ensemble classifier.

        Leveraging Bagging [^1] is an improvement over the Oza Bagging algorithm.
        The bagging performance is leveraged by increasing the re-sampling.
        It uses a poisson distribution to simulate the re-sampling process.
        To increase re-sampling it uses a higher `w` value of the Poisson
        distribution (agerage number of events), 6 by default, increasing the
        input space diversity, by attributing a different range of weights to the
        data samples.

        To deal with concept drift, Leveraging Bagging uses the ADWIN algorithm to
        monitor the performance of each member of the enemble If concept drift is
        detected, the worst member of the ensemble (based on the error estimation
        by ADWIN) is replaced by a new (empty) classifier.

        Parameters
        ----------
        model
            The classifier to bag.
        n_models
            The number of models in the ensemble.
        w
            Indicates the average number of events. This is the lambda parameter
            of the Poisson distribution used to compute the re-sampling weight.
        adwin_delta
            The delta parameter for the ADWIN change detector.
        bagging_method
            The bagging method to use. Can be one of the following:<br/>
            * 'bag' - Leveraging Bagging using ADWIN.<br/>
            * 'me' - Assigns $weight=1$ if sample is misclassified,
              otherwise $weight=error/(1-error)$.<br/>
            * 'half' - Use resampling without replacement for half of the instances.<br/>
            * 'wt' - Resample without taking out all instances.<br/>
            * 'subag' - Resampling without replacement.<br/>
        seed
            Random number generator seed for reproducibility.

        Examples
        --------

        >>> from river import datasets
        >>> from river import ensemble
        >>> from river import evaluate
        >>> from river import linear_model
        >>> from river import metrics
        >>> from river import optim
        >>> from river import preprocessing

        >>> dataset = datasets.Phishing()

        >>> model = ensemble.LeveragingBaggingClassifier(
        ...     model=(
        ...         preprocessing.StandardScaler() |
        ...         linear_model.LogisticRegression()
        ...     ),
        ...     n_models=3,
        ...     seed=42
        ... )

        >>> metric = metrics.F1()

        >>> evaluate.progressive_val_score(dataset, model, metric)
        F1: 0.886282

        """

    _BAGGING_METHODS = ("bag", "me", "half", "wt", "subag")

    def __init__(
            self,
            model: base.Classifier,
            param_grid,
            population_size=10,
            sampling_size=1,
            metric=metrics.Accuracy,
            sampling_rate=1000,
            w: float = 6,
            adwin_delta: float = 0.002,
            bagging_method: str = "bag",
            seed: int = None,
    ):

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
        self._population_metrics = [copy.deepcopy(metric()) for _ in range(self.n_models)]
        self._drift_detectors = [copy.deepcopy(ADWIN(delta=adwin_delta)) for _ in range(self.n_models)]
        self.n_detected_changes = 0
        self.w = w
        self.adwin_delta = adwin_delta
        self.bagging_method = bagging_method

        # Set bagging function
        if bagging_method == "bag":
            self._bagging_fct = self._leveraging_bag
        elif bagging_method == "me":
            self._bagging_fct = self._leveraging_bag_me
        elif bagging_method == "half":
            self._bagging_fct = self._leveraging_bag_half
        elif bagging_method == "wt":
            self._bagging_fct = self._leveraging_bag_wt
        elif bagging_method == "subag":
            self._bagging_fct = self._leveraging_subag
        else:
            raise ValueError(
                f"Invalid bagging_method: {bagging_method}\n"
                f"Valid options: {self._BAGGING_METHODS}"
            )

    def _initialize_model(self,model:base.Estimator,params):
        model = copy.deepcopy(model)
        model._set_params(params)
        return model

    def _leveraging_bag(self, **kwargs):
        # Leveraging bagging
        return self._rng.poisson(self.w)

    def _leveraging_bag_me(self, **kwargs):
        # Miss-classification error using weight=1 if misclassified.
        # Otherwise using weight=error/(1-error)
        x = kwargs["x"]
        y = kwargs["y"]
        i = kwargs["model_idx"]
        error = self._drift_detectors[i].estimation
        y_pred = self.models[i].predict_one(x)
        if y_pred != y:
            k = 1
        elif error != 1.0 and self._rng.rand() < (error / (1.0 - error)):
            k = 1
        else:
            k = 0
        return k

    def _leveraging_bag_half(self, **kwargs):
        # Resampling without replacement for half of the instances
        return int(not self._rng.randint(2))

    def _leveraging_bag_wt(self, **kwargs):
        # Resampling without taking out all instances
        return 1 + self._rng.poisson(1.0)

    def _leveraging_subag(self, **kwargs):
        # Subagging using resampling without replacement
        return int(self._rng.poisson(1) > 0)

    @property
    def bagging_methods(self):
        """Valid bagging_method options."""
        return self._BAGGING_METHODS


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

        change_detected = False
        for i, model in enumerate(self):
            self._population_metrics[i].update(y_true=y, y_pred=model.predict_one(x))
            k = self._bagging_fct(x=x, y=y, model_idx=i)

            for _ in range(k):
                model.learn_one(x, y)

            y_pred = self.models[i].predict_one(x)
            if y_pred is not None:
                incorrectly_classifies = int(y_pred != y)
                error = self._drift_detectors[i].estimation
                self._drift_detectors[i].update(incorrectly_classifies)
                if self._drift_detectors[i].change_detected:
                    if self._drift_detectors[i].estimation > error:
                        change_detected = True

        if change_detected:
            self.n_detected_changes += 1
            max_error_idx = max(
                range(len(self._drift_detectors)),
                key=lambda j: self._drift_detectors[j].estimation,
            )
            self.models[max_error_idx] = copy.deepcopy(self.model)
            self._drift_detectors[max_error_idx] = ADWIN(delta=self.adwin_delta)

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

    @classmethod
    def _unit_test_params(cls):
        models = [('HT',tree.HoeffdingTreeClassifier())]
        yield {
            "models": models,
        }

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

