import typing
import pandas as pd
from river import base
import numpy as np
from river.utils import dict2numpy
from tpot import TPOTClassifier


class OnlineTpotClassifer(base.Classifier):

    def __init__(self, n_training_samples, classes: list, max_time_mins: int = 15):
        self.n_training_samples = n_training_samples
        self.max_time_mins = max_time_mins
        self.training_samples_x = []
        self.training_samples_y = []
        self.estimator = None
        self.classes = classes

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        if self.estimator is None:
            self.training_samples_x.append(x)
            self.training_samples_y.append(y)
        if len(self.training_samples_x) >= self.n_training_samples and self.estimator is None:
            x_train = np.stack([dict2numpy(i) for i in self.training_samples_x])
            self.estimator = TPOTClassifier(max_time_mins=self.max_time_mins)
            self.estimator.fit(x_train, self.training_samples_y)
            self.training_samples_y = []
            self.training_samples_x = []
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.estimator is not None:
            y_pred = self.estimator.predict_proba([list(x.values())])[0]
            return {self.classes[i]: p for i, p in enumerate(y_pred)}
        else:
            return {c: 1 / len(self.classes) for c in self.classes}

    def predict_proba_many(self, X):
        return pd.Series(self.estimator.predict_proba(X), columns=self.classes)

    @property
    def _multiclass(self):
        return True

    def learn_many(self, X, y):
        self.estimator.partial_fit(X=X.values, y=y.values, classes=self.classes)
        return self

    def predict_one(self, x):
        if self.estimator is not None:
            y_pred = self.estimator.predict([list(x.values())])[0]
            return y_pred
        else:
            return self.classes[0]

    def predict_many(self, X):
        return pd.Series(self.estimator.predict(X))