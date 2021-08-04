import statistics

from river import base

from EvOAutoML.base.estimator import EvolutionaryBestEstimator


class EvolutionaryBestRegressor(EvolutionaryBestEstimator, base.Regressor):

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """Averages the predictions of each regressor."""
        arr = [regressor.predict_one(x) for regressor in self.population]
        return statistics.mean([.0 if v is None else v for v in arr])
