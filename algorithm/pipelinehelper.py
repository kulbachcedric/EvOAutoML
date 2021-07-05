"""Selects elements from a scikit pipeline with a working parametergrid."""
import copy
import random
import typing
from collections import defaultdict

import pandas as pd
from river import base
from river.base import Estimator, Classifier, Transformer, Regressor
from sklearn.model_selection import ParameterGrid
from sklearn.utils.metaestimators import if_delegate_has_method


class PipelineHelper(Estimator):

    def __init__(self,
                 available_models=None,
                 selected_model=None,
                 include_bypass=False,
                 optional=False
                 ):

        self.selected_model = selected_model
        self.include_bypass = include_bypass
        self.optional = optional
        # cloned
        if type(available_models) == dict:
            self.available_models = available_models
        else:
            # manually initialized
            self.available_models = {}
            for (key, model) in available_models:
                self.available_models[key] = model

    def clone(self):
        return copy.deepcopy(self)

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

        if self.optional:
            ret.append((None, dict()))
        return ret

    def _get_params(self, deep=True):
        """
        Returns the parameters of the current TransformerPicker instance.
        Note that this is different from the parameters used by the selected
        model. Provided for scikit estimator compatibility.
        """
        result = {
            'available_models': self.available_models,
            'selected_model': self.selected_model,
            'optional': self.optional,
        }
        if deep and self.selected_model:
            result.update({
                'selected_model__' + k: v
                for k, v in self.selected_model.get_params(deep=True).items()
            })
        if deep and self.available_models:
            for name, model in self.available_models.items():
                result['available_models__' + name] = model
                result.update({
                    'available_models__' + name + '__' + k: v
                    for k, v in model.get_params(deep=True).items()
                })
        return result

    def _set_params(self, new_params: dict = None):
        """
        Sets the parameters to all available models.
        Provided for scikit estimator compatibility.
        """
        if len(new_params) > 0:
            self.selected_model = self.available_models[new_params[0]]
            self.selected_model._set_params(new_params=new_params[1])
        else:
            self.selected_model = self.available_models[random.choice(list(self.available_models))]
        return self



class PipelineHelperClassifier(PipelineHelper,Classifier):

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> Estimator:
        """Fits the selected model."""
        if self.selected_model is None or self.selected_model == 'passthrough':
            return self
        else:
            return self.selected_model.learn_one(x=x, y=y, **kwargs)

    def predict_one(self, x: dict):
        if self.selected_model is None or self.selected_model == 'passthrough':
            return self
        else:
            return self.selected_model.predict_one(x)

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.selected_model is None or self.selected_model == 'passthrough':
            return self
        else:
            return self.selected_model.predict_proba_one(x)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_model is None or self.selected_model == 'passthrough':
            return self
        else:
            return self.selected_model.predict_proba_many(X=X)

class PipelineHelperTransformer(PipelineHelper, Transformer):

    def transform_one(self, x: dict) -> dict:
        if self.selected_model is None or self.selected_model == 'passthrough':
            return x
        else:
            return self.selected_model.transform_one(x=x)