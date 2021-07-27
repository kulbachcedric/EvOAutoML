import collections
import statistics
import typing
from river import base

from EvOAutoML.base import EvolutionaryBestEstimator


class EvolutionaryBestClassifier(EvolutionaryBestEstimator, base.Classifier):

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        """Averages the predictions of each classifier."""
        y_pred = collections.Counter()
        for classifier in self.population:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred


class EvolutionaryBestRegressor(EvolutionaryBestEstimator, base.Regressor):

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """Averages the predictions of each regressor."""
        return statistics.mean((regressor.predict_one(x) for regressor in self.population))