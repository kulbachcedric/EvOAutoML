import collections

from river import base, metrics, tree

from EvOAutoML.base.evolution import (
    EvolutionaryBaggingEstimator,
    EvolutionaryBaggingOldestEstimator,
)
from EvOAutoML.config import (
    AUTOML_CLASSIFICATION_PIPELINE,
    CLASSIFICATION_PARAM_GRID,
)


class EvolutionaryBaggingClassifier(
    EvolutionaryBaggingEstimator, base.Classifier
):
    """
    Evolutionary Bagging Classifier follows the Oza Bagging approach to update
    the population of estimator pipelines.

    Parameters
    ----------
    model
        A river model or model pipeline that can be configured
        by the parameter grid.
    param_grid
        A parameter grid, that represents the configuration space of the model.
    population_size
        The population size estimates the size of the population as
        well as the size of the ensemble used for the prediction.
    sampling_size
        The sampling size estimates how many models are mutated
        within one mutation step.
    metric
        The river metric that should be optimised.
    sampling_rate
        The sampling rate estimates the number of samples that are executed
        before a mutation step takes place.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------
    >>> from river import datasets, ensemble, evaluate, metrics, compose, optim
    >>> from river import preprocessing, neighbors, naive_bayes, tree
    >>> from river import linear_model
    >>> from EvOAutoML import classification, pipelinehelper
    >>> dataset = datasets.Phishing()
    >>> model = classification.EvolutionaryBaggingClassifier(seed=42)
    >>> metric = metrics.F1()
    >>> for x, y in dataset:
    ...     y_pred = model.predict_one(x)  # make a prediction
    ...     metric = metric.update(y, y_pred)  # update the metric
    ...     model = model.learn_one(x,y)  # make the model learn
    """

    def __init__(
        self,
        model=AUTOML_CLASSIFICATION_PIPELINE,
        param_grid=CLASSIFICATION_PARAM_GRID,
        population_size=10,
        sampling_size=1,
        metric=metrics.Accuracy,
        sampling_rate=1000,
        seed=42,
    ):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            sampling_rate=sampling_rate,
            seed=seed,
        )

    @classmethod
    def _unit_test_params(cls):
        model = tree.HoeffdingTreeClassifier()

        param_grid = {
            "max_depth": [10, 30, 60, 10, 30, 60],
        }

        yield {
            "model": model,
            "param_grid": param_grid,
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {"check_init_default_params_are_not_mutable"}

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred


class EvolutionaryOldestBaggingClassifier(
    EvolutionaryBaggingOldestEstimator, base.Classifier
):
    """
    Evolutionary Oldest Bagging Classifier follows the Oza Bagging approach
    to update the population of estimator pipelines. It mutates the population
    by removing the oldest model configuration.

    Parameters
    ----------
    model
        A river model or model pipeline that can be configured
        by the parameter grid.
    param_grid
        A parameter grid, that represents the configuration space of the model.
    population_size
        The population size estimates the size of the population as
        well as the size of the ensemble used for the prediction.
    sampling_size
        The sampling size estimates how many models are mutated
        within one mutation step.
    metric
        The river metric that should be optimised.
    sampling_rate
        The sampling rate estimates the number of samples that are executed
        before a mutation step takes place.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------
    >>> from river import datasets, ensemble, evaluate, metrics, compose, optim
    >>> from river import preprocessing, neighbors, naive_bayes, tree
    >>> from river import linear_model
    >>> from EvOAutoML import classification, pipelinehelper
    >>> dataset = datasets.Phishing()
    >>> model = classification.EvolutionaryOldestBaggingClassifier(seed=42)
    >>> metric = metrics.F1()
    >>> for x, y in dataset:
    ...     y_pred = model.predict_one(x)  # make a prediction
    ...     metric = metric.update(y, y_pred)  # update the metric
    ...     model = model.learn_one(x,y)  # make the model learn
    """

    def __init__(
        self,
        model=AUTOML_CLASSIFICATION_PIPELINE,
        param_grid=CLASSIFICATION_PARAM_GRID,
        population_size=10,
        sampling_size=1,
        metric=metrics.Accuracy,
        sampling_rate=1000,
        seed=42,
    ):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            sampling_rate=sampling_rate,
            seed=seed,
        )

    @classmethod
    def _unit_test_params(cls):
        model = tree.HoeffdingTreeClassifier()

        param_grid = {
            "max_depth": [10, 30, 60, 10, 30, 60],
        }

        yield {
            "model": model,
            "param_grid": param_grid,
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {"check_init_default_params_are_not_mutable"}

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred
