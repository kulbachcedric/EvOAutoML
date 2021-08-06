from multiprocessing import Pool
from pathlib import Path

from river import ensemble

from EvOAutoML.classification import EvolutionaryBaggingClassifier, EvolutionaryLeveragingBaggingClassifer
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    ENSEMBLE_CLASSIFIER, POPULATION_SIZE, N_SAMPLES, N_CHECKPOINTS, SAMPLING_RATE
from EvOAutoML.utils import plot_track


def evaluate_ensemble(track_tuple):
    track = track_tuple[1]
    plot_track(
        track=track,
        metric_name="Accuracy",
        models={
            'EvoAutoML Bagging': EvolutionaryBaggingClassifier(population_size=POPULATION_SIZE, model=AUTOML_CLASSIFICATION_PIPELINE,
                                                       param_grid=CLASSIFICATION_PARAM_GRID, sampling_rate=SAMPLING_RATE),
            #'EvoAutoML Leveraging': EvolutionaryLeveragingBaggingClassifer(population_size=POPULATION_SIZE,
            #                                                             model=AUTOML_CLASSIFICATION_PIPELINE,
            #                                                             param_grid=CLASSIFICATION_PARAM_GRID,
            #                                                             sampling_rate=SAMPLING_RATE),
            # 'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
            ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
            'Ada Boost': ensemble.AdaBoostClassifier(model=ENSEMBLE_CLASSIFIER()),
            'ARF': ensemble.AdaptiveRandomForestClassifier(),
            'Leveraging Bagging': ensemble.LeveragingBaggingClassifier(model=ENSEMBLE_CLASSIFIER()),
            'Bagging' : ensemble.BaggingClassifier(model=ENSEMBLE_CLASSIFIER()),
        },
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        result_path=Path(f'./results/classification/evaluation_ensemble'),
        verbose=2
    )


if __name__ == '__main__':

    #evaluate_ensemble(CLASSIFICATION_TRACKS[0])
    pool = Pool(40)  # Create a multiprocessing Pool
    output = pool.map(evaluate_ensemble, CLASSIFICATION_TRACKS)