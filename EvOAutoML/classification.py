import collections
import typing

from river import base

from EvOAutoML.base.estimator import EvolutionaryBestEstimator


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

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        """Predict the label of a set of features `x`.

                Parameters
                ----------
                x
                    A dictionary of features.

                Returns
                -------
                The predicted label.

                """

        # The following code acts as a default for each classifier, and may be overridden on an
        # individual basis.
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None
