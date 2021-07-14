"""Selects elements from a scikit pipeline with a working parametergrid."""
import collections
import copy
import random
import typing
from collections import defaultdict

import pandas as pd
from river import base
from river.base import Estimator, Classifier, Transformer, Regressor, EnsembleMixin
from sklearn.model_selection import ParameterGrid
from sklearn.utils.metaestimators import if_delegate_has_method




class PipelineHelper(Estimator):
    def __init__(self, models, selected_model=None):
        self.selected_model = None

        # cloned
        if type(models) == dict:
            self.models = models
        else:
            # manually initialized
            self.available_models = {}
            for (key, model) in models:
                self.available_models[key] = model

        if selected_model is None:
            self.selected_model = self.available_models[random.choice(list(self.available_models))]
        else:
            self.selected_model = selected_model

        #super().__init__(models)

    def clone(self):
        return PipelineHelper(self.models, self.selected_model)

    def generate(self, param_dict=None):
        """
        Generates the parameters that are required for a search.
        Args:
            param_dict: parameters for the available models provided in the
                constructor. Note that these don't require the prefix path of
                all elements higher up the hierarchy of this TransformerPicker.
        """
        if param_dict is None:
            param_dict = dict()
        per_model_parameters = defaultdict(lambda: defaultdict(list))

        # collect parameters for each specified model
        for k, values in param_dict.items():
            # example:  randomforest__n_estimators
            model_name = k.split('__')[0]
            param_name = k[len(model_name) + 2:]
            if model_name not in self.available_models:
                raise Exception('no such model: {0}'.format(model_name))
            per_model_parameters[model_name][param_name] = values

        ret = []

        # create instance for cartesion product of all available parameters
        # for each model
        for model_name, param_dict in per_model_parameters.items():
            parameter_sets = ParameterGrid(param_dict)
            for parameters in parameter_sets:
                ret.append((model_name, parameters))

        # for every model that has no specified parameters, add default value
        for model_name in self.available_models.keys():
            if model_name not in per_model_parameters:
                ret.append((model_name, dict()))

        return ret

    def _get_params(self, deep=True):
        return self.selected_model._get_params()

    def _set_params(self, new_params: dict = None):
        """
        Sets the parameters to all available models.
        Provided for scikit estimator compatibility.
        """
        if len(new_params) > 0:
            self.selected_model = self.available_models[new_params[0]]
            self.selected_model._set_params(new_params=new_params[1])
        elif self.selected_model == None:
            self.selected_model = self.available_models[random.choice(list(self.available_models))]
        return self



class PipelineHelperClassifier(PipelineHelper,Classifier):

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> Estimator:
        """Fits the selected model."""
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
    @property
    def _supervised(self):
        if self.selected_model._supervised:
            return True
        else:
            return False

    def transform_one(self, x: dict) -> dict:
        return self.selected_model.transform_one(x=x)

    def learn_one(self, x: dict, y: base.typing.Target = None, **kwargs) -> "Transformer":
        """Update with a set of features `x`.

        A lot of transformers don't actually have to do anything during the `learn_one` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_one` can override this
        method.

        Parameters
        ----------
        x
            A dictionary of features.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """
        if self.selected_model._supervised:
            self.selected_model = self.selected_model.learn_one(x, y)
        else:
            self.selected_model = self.selected_model.learn_one(x)
        return self


