import typing

import pandas as pd
from river import base, preprocessing, tree
from river.base import Classifier, Estimator, Regressor, Transformer

from EvOAutoML.base.utils import PipelineHelper


class PipelineHelperClassifier(PipelineHelper, Classifier):
    """
    This class is used to create a pipeline, where multiple classifiers are
    able to used in parallel. The selected classifier (`selected_model`) is
    used to make predictions as well as for training.
    The other classifiers are not trained in parallel.

    Parameters
    ----------
    models: dict
        A dictionary of models that can be used in the pipeline.
    selected_model: Estimator
        the model that is used for training and prediction.
    """
    @classmethod
    def _unit_test_params(cls):
        models = [
            ("HT", tree.HoeffdingTreeClassifier()),
            ("EFDT", tree.ExtremelyFastDecisionTreeClassifier()),
        ]
        yield {
            "models": models,
        }

    def learn_one(
        self, x: dict, y: base.typing.ClfTarget, **kwargs
    ) -> Estimator:
        self.selected_model = self.selected_model.learn_one(x=x, y=y, **kwargs)
        return self

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        return self.selected_model.predict_one(x)


    def predict_proba_one(
        self, x: dict
    ) -> typing.Dict[base.typing.ClfTarget, float]:
        return self.selected_model.predict_proba_one(x=x)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.selected_model.predict_proba_many(X=X)

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        return self.selected_model.predict_many(X=X)


class PipelineHelperTransformer(PipelineHelper, Transformer):
    """
    Add some Text here
    """

    @classmethod
    def _unit_test_params(cls):
        models = [
            ("ABS", preprocessing.MaxAbsScaler()),
            ("NORM", preprocessing.Normalizer()),
        ]
        yield {
            "models": models,
        }

    @property
    def _supervised(self):
        if self.selected_model._supervised:
            return True
        else:
            return False

    def transform_one(self, x: dict) -> dict:
        """

        Args:
            x:

        Returns:

        """
        return self.selected_model.transform_one(x=x)

    def learn_one(
        self, x: dict, y: base.typing.Target = None, **kwargs
    ) -> "Transformer":
        """
        Add second text here
        Args:
            x:
            y:
            **kwargs:

        Returns:
            self
        """
        if self.selected_model._supervised:
            self.selected_model = self.selected_model.learn_one(x, y)
        else:
            self.selected_model = self.selected_model.learn_one(x)
        return self


class PipelineHelperRegressor(PipelineHelper, Regressor):
    @classmethod
    def _unit_test_params(cls):
        models = [
            ("HT", tree.HoeffdingTreeRegressor()),
            ("HAT", tree.HoeffdingAdaptiveTreeRegressor()),
        ]
        yield {
            "models": models,
        }

    def learn_one(
        self, x: dict, y: base.typing.ClfTarget, **kwargs
    ) -> Estimator:
        self.selected_model = self.selected_model.learn_one(x=x, y=y, **kwargs)
        return self

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        return self.selected_model.predict_one(x=x)
