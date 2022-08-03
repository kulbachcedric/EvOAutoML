import statistics

from river import base, metrics

from EvOAutoML.base.estimator import EvolutionaryBaggingEstimator
from EvOAutoML.config import AUTOML_REGRESSION_PIPELINE, REGRESSION_PARAM_GRID


class EvolutionaryBaggingRegressor(EvolutionaryBaggingEstimator, base.Regressor):

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
