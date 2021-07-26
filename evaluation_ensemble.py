from pathlib import Path

from river import tree, ensemble
from tqdm import tqdm

from EvOAutoML.oaml import EvolutionaryBestClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID

from EvOAutoML.utils import plot_track

if __name__ == '__main__':

    ensemble_model = tree.HoeffdingTreeClassifier()
    for track_name, track in tqdm(CLASSIFICATION_TRACKS):
        fig = plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=AUTOML_CLASSIFICATION_PIPELINE, param_grid=CLASSIFICATION_PARAM_GRID, sampling_rate=100),
                #'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
                ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
                'Bagging' : ensemble.BaggingClassifier(model=ensemble_model),
                'Ada Boost' : ensemble.AdaBoostClassifier(model=ensemble_model),
                'ARFC' : ensemble.AdaptiveRandomForestClassifier(),
                'LB' : ensemble.LeveragingBaggingClassifier(model=ensemble_model),
                'Adwin Bagging' : ensemble.ADWINBaggingClassifier(model=ensemble_model),
            },
            n_samples=10_000,
            n_checkpoints=1000,
            result_path=Path(f'./results/evaluation_ensemble'),
            verbose=2
        )