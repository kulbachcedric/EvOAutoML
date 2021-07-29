from pathlib import Path

from river import tree, ensemble
from tqdm import tqdm

from EvOAutoML.classification import EvolutionaryBestClassifier
#from EvOAutoML.ray_classification import DecentralizedEvolutionaryBestClassifier

from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    ENSEMBLE_CLASSIFIER
from EvOAutoML.ray_classification import DecentralizedEvolutionaryBestClassifier
from EvOAutoML.thread_classification import ThreadEvolutionaryBestClassifier

from EvOAutoML.utils import plot_track

if __name__ == '__main__':
    ensemble_model = tree.HoeffdingTreeClassifier()
    population_size = 10

    for track_name, track in tqdm(CLASSIFICATION_TRACKS):
        fig = plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                #'ThreadAutoML': ThreadEvolutionaryBestClassifier(population_size=50,
                #                                                 estimator=AUTOML_CLASSIFICATION_PIPELINE,
                #                                                 param_grid=CLASSIFICATION_PARAM_GRID,
                #                                                 sampling_rate=250),
                'Decentralized EvoAutoML': DecentralizedEvolutionaryBestClassifier(population_size=50,
                                                                                   estimator=AUTOML_CLASSIFICATION_PIPELINE,
                                                                                   param_grid=CLASSIFICATION_PARAM_GRID,
                                                                                   sampling_rate=250),
                'EvoAutoML': EvolutionaryBestClassifier(population_size=50,
                                                        estimator=AUTOML_CLASSIFICATION_PIPELINE,
                                                        param_grid=CLASSIFICATION_PARAM_GRID,
                                                        sampling_rate=250),
                'Ada Boost': ensemble.AdaBoostClassifier(model=ensemble_model,n_models=50),
                'ARFC': ensemble.AdaptiveRandomForestClassifier(n_models=50),
                'LB': ensemble.LeveragingBaggingClassifier(model=ensemble_model,n_models=50),
            },
            n_samples=10_000,
            n_checkpoints=1000,
            result_path=Path(f'./results/evaluation_parallelization'),
            verbose=2
        )