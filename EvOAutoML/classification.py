import collections
import typing

import ray
from river import base

from EvOAutoML.base.estimator import EvolutionaryBaggingEstimator, EvolutionaryLeveragingBaggingEstimator


class EvolutionaryBaggingClassifier(EvolutionaryBaggingEstimator, base.Classifier):

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(ray.get(classifier.predict_proba_one.remote(x)))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

class EvolutionaryLeveragingBaggingClassifer(EvolutionaryLeveragingBaggingEstimator, base.Classifier):
    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(ray.get(classifier.predict_proba_one.remote(x)))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred