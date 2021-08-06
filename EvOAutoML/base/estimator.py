import copy
import random
from collections import defaultdict

import numpy as np
import ray
from river import base
from river import metrics
from river.base import Estimator
from river.drift import ADWIN
from river.metrics import ClassificationMetric
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

@ray.remote
class IndividualBagging(base.Estimator):
    def __init__(self, model, metric):
        self.model = copy.deepcopy(model)
        self.metric = metric()

    def learn_one(self,x,y,poi):
        for i in range(poi):
            self.model.learn_one(x,y)

    def set_model(self, model):
        self.model = copy.deepcopy(model)

    def get_model(self):
        return self.model

    def get_score(self):
        return self.metric.get()

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def predict_one(self,x):
        return self.model.predict_one(x)

    def update_metric(self,y_true, y_pred):
        self.metric.update(y_true, y_pred)

class EvolutionaryBaggingEstimator(base.WrapperMixin, base.EnsembleMixin):

    def __init__(self, model,
                 param_grid,
                 population_size=10,
                 sampling_size=1,
                 metric=metrics.Accuracy,
                 sampling_rate=1000,
                 seed=42):
        ray.init()
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

        param_iter = ParameterSampler(param_grid, population_size)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]
        super().__init__(self._initialize_model(model=model, params=params) for params in param_list)
        #self._drift_detectors = [copy.deepcopy(ADWIN()) for _ in range(self.n_models)]
        #self._population_metrics = [copy.deepcopy(metric()) for _ in range(self.n_models)]


    @property
    def _wrapped_model(self):
        return self.model


    def _initialize_model(self,model:base.Estimator,params):
        model = copy.deepcopy(model)
        model._set_params(params)
        return IndividualBagging.remote(model=model, metric=self.metric)

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        # Create Dataset if not initialized
        # Check if population needs to be updated
        if self._i % self.sampling_rate == 0:
            scores = [be.get_score.remote() for be in self]
            scores = ray.get(scores)
            idx_best = scores.index(max(scores))
            idx_worst = scores.index(min(scores))
            child = self._mutate_estimator(estimator=ray.get(self[idx_best].get_model.remote()))
            self.models[idx_worst].set_model.remote(child)
            #self.population_metrics[idx_worst] = copy.deepcopy(self.metric())

        for idx, model in enumerate(self):
            self[idx].update_metric.remote(y_true=y, y_pred=ray.get(model.predict_one.remote(x)))
            model.learn_one.remote(x, y, self._rng.poisson(5))
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

@ray.remote
class IndividualLeveragingBagging(base.Estimator):
    def __init__(self, model, metric, change_detector):
        self.model = copy.deepcopy(model)
        self.metric = metric()
        self.change_detector = copy.deepcopy(change_detector)

    def learn_one(self,x,y,poi):
        for i in range(poi):
            self.model.learn_one(x,y)

    def set_model(self, model):
        self.model = copy.deepcopy(model)

    def get_model(self):
        return self.model

    def get_score(self):
        return self.metric.get()

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def predict_one(self,x):
        return self.model.predict_one(x)

    def update_metric(self,y_true, y_pred):
        self.metric.update(y_true, y_pred)

    def update_change_detector(self,incorrectly_classifies):
        self.change_detector.update(incorrectly_classifies)

    def estimation(self):
        return self.change_detector.estimation

    def change_detected(self):
        return self.change_detector.change_detected

    def set_change_detector(self, change_detector):
        self.change_detector = copy.deepcopy(change_detector)

class EvolutionaryLeveragingBaggingEstimator(base.WrapperMixin, base.EnsembleMixin):
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
        #self._population_metrics = [copy.deepcopy(metric()) for _ in range(self.n_models)]
        #self._drift_detectors = [copy.deepcopy(ADWIN(delta=adwin_delta)) for _ in range(self.n_models)]
        self.n_detected_changes = 0
        self.w = w
        self.adwin_delta = adwin_delta
        self.bagging_method = bagging_method

        param_iter = ParameterSampler(param_grid, population_size)
        param_list = list(param_iter)
        param_list = [dict((k, v) for (k, v) in d.items()) for d in
                      param_list]
        super().__init__(self._initialize_model(model=model,params=params) for params in param_list)

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
        return IndividualLeveragingBagging.remote(model=model,metric=self.metric,change_detector=ADWIN(delta=self.adwin_delta))

    def _leveraging_bag(self, **kwargs):
        # Leveraging bagging
        return self._rng.poisson(self.w)

    def _leveraging_bag_me(self, **kwargs):
        # Miss-classification error using weight=1 if misclassified.
        # Otherwise using weight=error/(1-error)
        x = kwargs["x"]
        y = kwargs["y"]
        i = kwargs["model_idx"]
        error = self[i].remote.estimation
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
            scores = [be.get_score.remote() for be in self]
            scores = ray.get(scores)
            idx_best = scores.index(max(scores))
            idx_worst = scores.index(min(scores))
            child = self._mutate_estimator(estimator=ray.get(self[idx_best].get_model.remote()))
            self[idx_worst].set_model.remote(child)
            #self.population_metrics[idx_worst] = copy.deepcopy(self.metric())

        change_detected = False
        for i, model in enumerate(self):
            #self._population_metrics[i].update(y_true=y, y_pred=model.predict_one(x))
            self[i].update_metric.remote(y_true=y,y_pred=ray.get(self[i].predict_one.remote(x)))
            k = self._bagging_fct(x=x, y=y, model_idx=i)

            model.learn_one.remote(x, y, k)

            y_pred = ray.get(self[i].predict_one.remote(x))
            if y_pred is not None:
                incorrectly_classifies = int(y_pred != y)
                error = self[i].estimation.remote()
                self[i].update_change_detector.remote(incorrectly_classifies)
                if ray.get(self[i].change_detected.remote()):
                    if ray.get(self[i].estimation.remote()) > ray.get(error):
                        change_detected = True

        if change_detected:
            self.n_detected_changes += 1
            max_error_idx = max(
                range(self.population_size),
                key=lambda j: ray.get(self[j].estimation.remote()),
            )
            self[max_error_idx].set_model.remote(self.model) #todo change model correctly
            self[max_error_idx].set_change_detector.remote(ADWIN(delta=self.adwin_delta))

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

