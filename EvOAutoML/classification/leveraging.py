import collections
import typing

from river import base, metrics

from EvOAutoML.base.evolution import EvolutionaryLeveragingBaggingEstimator
from EvOAutoML.config import AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID

class EvolutionaryLeveragingBaggingClassifer(EvolutionaryLeveragingBaggingEstimator, base.Classifier):
    """
    Evolutionary Leveraging Bagging Classifier follows a Leveraging Bagging approach to update the population of
    estimator pipelines.

    Parameters
    ----------
    model
        A model pipeline
    param_grid
    population_size
    sampling_size
    metric
    sampling_rate
    w
    adwin_delta
    bagging_method
    seed
    """
    def __init__(
            self,
            model: base.Classifier = AUTOML_CLASSIFICATION_PIPELINE,
            param_grid=CLASSIFICATION_PARAM_GRID,
            population_size:int=10,
            sampling_size:int=1,
            metric=metrics.Accuracy,
            sampling_rate:int =1000,
            w:int = 6,
            adwin_delta: float = 0.002,
            bagging_method: str = "bag",
            seed: int = 42,
    ):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            sampling_rate=sampling_rate,
            w=w,
            adwin_delta=adwin_delta,
            bagging_method=bagging_method,
            seed=seed
        )
    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred