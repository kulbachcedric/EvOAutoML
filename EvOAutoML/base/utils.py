import random
from collections import defaultdict

from river import tree
from river.base import Estimator
from sklearn.model_selection import ParameterGrid


class PipelineHelper(Estimator):
    @classmethod
    def _unit_test_params(cls):
        models = [("HT", tree.HoeffdingTreeClassifier())]
        yield {
            "models": models,
        }

    def __init__(self, models, selected_model=None):
        self.selected_model = None
        self.models = None

        # cloned
        if type(models) == dict:
            self.models = models
        else:
            self.available_models = {}
            for (key, model) in models:
                self.available_models[key] = model

        if selected_model is None:
            self.selected_model = self.available_models[
                random.choice(list(self.available_models))
            ]
        else:
            self.selected_model = selected_model

    def clone(self):
        return PipelineHelper(self.models)  # , self.selected_model.clone())

    def generate(self, param_dict=None):
        if param_dict is None:
            param_dict = dict()
        per_model_parameters = defaultdict(lambda: defaultdict(list))

        # collect parameters for each specified model
        for k, values in param_dict.items():
            # example:  randomforest__n_estimators
            model_name = k.split("__")[0]
            param_name = k[len(model_name) + 2 :]
            if model_name not in self.available_models:
                raise Exception("no such model: {0}".format(model_name))
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
        if len(new_params) > 0:
            self.selected_model = self.available_models[new_params[0]].__class__(
                **new_params[1]
            )
        elif self.selected_model == None:
            self.selected_model = self.available_models[
                random.choice(list(self.available_models))
            ]
        return self
