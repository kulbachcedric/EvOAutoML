import collections
import typing

from river import base

from EvOAutoML.base.estimator import EvolutionaryBaggingEstimator, EvolutionaryLeveragingBaggingEstimator, \
    EvolutionaryBaggingOldestEstimator


class EvolutionaryBaggingClassifier(EvolutionaryBaggingEstimator, base.Classifier):

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

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

class EvolutionaryLeveragingBaggingClassifer(EvolutionaryLeveragingBaggingEstimator, base.Classifier):

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred