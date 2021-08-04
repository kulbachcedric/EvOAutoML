from multiprocessing import Pool
from pathlib import Path

from river import tree, ensemble
from river.linear_model import LinearRegression
from river.neighbors import KNNClassifier
from tqdm import tqdm

from EvOAutoML.classification import EvolutionaryBestClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    ENSEMBLE_CLASSIFIER, POPULATION_SIZE, AUTOML_REGRESSION_PIPELINE, REGRESSION_PARAM_GRID, ENSEMBLE_REGRESSOR, \
    REGRESSION_TRACKS, N_SAMPLES, N_CHECKPOINTS, SAMPLING_RATE
from EvOAutoML.regression import EvolutionaryBestRegressor

from EvOAutoML.utils import plot_track





def evaluate_ensemble(track_tuple):
    track = track_tuple[1]
    plot_track(
        track=track,
        metric_name="R2",
        models={
            'EvoAutoML': EvolutionaryBestRegressor(population_size=POPULATION_SIZE, estimator=AUTOML_REGRESSION_PIPELINE,
                                                    param_grid=REGRESSION_PARAM_GRID, sampling_rate=SAMPLING_RATE),
            # 'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
            ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),

            #'ARF': ensemble.AdaptiveRandomForestRegressor(),
            'Bagging': ensemble.BaggingRegressor(model=ENSEMBLE_REGRESSOR()),
        },
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        result_path=Path(f'./results/regression/evaluation_ensemble'),
        verbose=2
    )


if __name__ == '__main__':

    #evaluate_ensemble(REGRESSION_TRACKS[0])
    pool = Pool(40)  # Create a multiprocessing Pool
    output = pool.map(evaluate_ensemble, REGRESSION_TRACKS)