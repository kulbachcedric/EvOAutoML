import statistics

from river import base, metrics

from EvOAutoML.base.evolution import EvolutionaryBaggingEstimator
from EvOAutoML.config import AUTOML_REGRESSION_PIPELINE, REGRESSION_PARAM_GRID


class EvolutionaryBaggingRegressor(EvolutionaryBaggingEstimator, base.Regressor):
    """
    Evolutionary Bagging Regressor follows the Oza Bagging approach to update
    the population of estimator pipelines.

    Parameters
    ----------
    model
        A river model or model pipeline that can be configured
        by the parameter grid.
    param_grid
        A parameter grid, that represents the configuration space of the model.
    population_size
        The population size estimates the size of the population as
        well as the size of the ensemble used for the prediction.
    sampling_size
        The sampling size estimates how many models are mutated
        within one mutation step.
    metric
        The river metric that should be optimised.
    sampling_rate
        The sampling rate estimates the number of samples that are executed
        before a mutation step takes place.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------
    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing
    >>> dataset = datasets.TrumpApproval()
    >>> model = ensemble.EvolutionaryBaggingClassifer(
    ...     model=compose.Pipeline(
    ...         ('Scaler', PipelineHelperTransformer([
    ...             ('StandardScaler', preprocessing.StandardScaler()),
    ...             ('MinMaxScaler', preprocessing.MinMaxScaler()),
    ...             ('MinAbsScaler', preprocessing.MaxAbsScaler()),
    ...         ])),
    ...         ('Regressor', PipelineHelperRegressor([
    ...             ('HT', tree.HoeffdingTreeRegressor()),
    ...             ('KNN', neighbors.KNNRegressor()),
    ...         ]))
    ...         ),
    ...     param_grid={
    ...     'Regressor': AUTOML_REGRESSION_PIPELINE.steps['Regressor'].generate({
    ...         'HT__binary_split': [True, False],
    ...         'HT__max_depth': [10, 30, 60, 10, 30, 60],
    ...         'HT__grace_period': [10, 100, 200, 10, 100, 200],
    ...         'HT__max_size': [5, 10],
    ...         'KNN__n_neighbors': [1, 5, 20],
    ...         'KNN__window_size': [100, 500, 1000],
    ...         'KNN__p': [1, 2]
    ...     })
    ... }
    ... seed=42
    ... )
    >>> metric = metrics.MSE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MSE: 88.73%
    """
    def __init__(self,
                 model=AUTOML_REGRESSION_PIPELINE,
                 param_grid=REGRESSION_PARAM_GRID,
                 population_size=10,
                 sampling_size=1,
                 metric= metrics.MSE,
                 sampling_rate=1000,
                 seed=42):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            sampling_rate=sampling_rate,
            seed=seed
        )


    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """Averages the predictions of each regressor."""
        arr = [regressor.predict_one(x) for regressor in self]
        return statistics.mean([.0 if v is None else v for v in arr])
