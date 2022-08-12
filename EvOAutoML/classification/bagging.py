import collections
import typing

from river import base, metrics

from EvOAutoML.base.evolution import EvolutionaryBaggingEstimator, EvolutionaryBaggingOldestEstimator
from EvOAutoML.config import AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID

class EvolutionaryBaggingClassifier(EvolutionaryBaggingEstimator, base.Classifier):

    def __init__(self,
                 model=AUTOML_CLASSIFICATION_PIPELINE,
                 param_grid=CLASSIFICATION_PARAM_GRID,
                 population_size=10,
                 sampling_size=1,
                 metric= metrics.Accuracy,
                 sampling_rate=1000,
                 seed=42):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            sampling_rate=sampling_rate,
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

class EvolutionaryOldestBaggingClassifier(EvolutionaryBaggingOldestEstimator, base.Classifier):


    def __init__(self,
                 model=AUTOML_CLASSIFICATION_PIPELINE,
                 param_grid=CLASSIFICATION_PARAM_GRID,
                 population_size=10,
                 sampling_size=1,
                 metric= metrics.Accuracy,
                 sampling_rate=1000,
                 seed=42):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            sampling_rate=sampling_rate,
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