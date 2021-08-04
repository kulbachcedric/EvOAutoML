import typing

import pandas as pd
from river import base
from river.base import Estimator, Classifier, Transformer, Regressor

from EvOAutoML.base.estimator import PipelineHelper


class PipelineHelperClassifier(PipelineHelper,Classifier):

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> Estimator:
        self.selected_model = self.selected_model.learn_one(x=x, y=y, **kwargs)
        return self

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        return self.selected_model.predict_proba_one(x=x)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.selected_model.predict_proba_many(X=X)

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.predict_proba_many(X)
        if y_pred.empty:
            return y_pred
        return y_pred.idxmax(axis="columns")

class PipelineHelperTransformer(PipelineHelper, Transformer):
    """
    Add some Text here
    """
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

    def learn_one(self, x: dict, y: base.typing.Target = None, **kwargs) -> "Transformer":
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

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> Estimator:
        self.selected_model = self.selected_model.learn_one(x=x, y=y, **kwargs)
        return self

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        return self.selected_model.predict_one(x=x)