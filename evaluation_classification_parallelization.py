from pathlib import Path

from river import ensemble
from tqdm import tqdm

from EvOAutoML.classification import EvolutionaryBestClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    ENSEMBLE_CLASSIFIER, POPULATION_SIZE, N_CHECKPOINTS, N_SAMPLES, SAMPLING_RATE
from EvOAutoML.ray_classification import DecentralizedEvolutionaryBestClassifier
from EvOAutoML.thread_classification import ThreadEvolutionaryBestClassifier
from EvOAutoML.utils import plot_track

if __name__ == '__main__':

    for track_name, track in tqdm(CLASSIFICATION_TRACKS):
        fig = plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                'ThreadAutoML': ThreadEvolutionaryBestClassifier(population_size=POPULATION_SIZE,
                                                                 estimator=AUTOML_CLASSIFICATION_PIPELINE,
                                                                 param_grid=CLASSIFICATION_PARAM_GRID,
                                                                 sampling_rate=SAMPLING_RATE),
                'Decentralized EvoAutoML': DecentralizedEvolutionaryBestClassifier(population_size=POPULATION_SIZE,
                                                                                   estimator=AUTOML_CLASSIFICATION_PIPELINE,
                                                                                   param_grid=CLASSIFICATION_PARAM_GRID,
                                                                                   sampling_rate=SAMPLING_RATE),
                'EvoAutoML': EvolutionaryBestClassifier(population_size=POPULATION_SIZE,
                                                        estimator=AUTOML_CLASSIFICATION_PIPELINE,
                                                        param_grid=CLASSIFICATION_PARAM_GRID,
                                                        sampling_rate=SAMPLING_RATE),
                'Ada Boost': ensemble.AdaBoostClassifier(model=ENSEMBLE_CLASSIFIER(),n_models=POPULATION_SIZE),
                'ARFC': ensemble.AdaptiveRandomForestClassifier(n_models=POPULATION_SIZE),
                'LB': ensemble.LeveragingBaggingClassifier(model=ENSEMBLE_CLASSIFIER(),n_models=POPULATION_SIZE),
            },
            n_samples=N_SAMPLES,
            n_checkpoints=N_CHECKPOINTS,
            result_path=Path(f'./results/classification/evaluation_parallelization'),
            verbose=2
        )