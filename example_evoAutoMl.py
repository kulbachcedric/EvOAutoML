from pathlib import Path

from river import tree, preprocessing, compose, naive_bayes, neighbors, ensemble, feature_extraction
from tqdm import tqdm

from algorithm.oaml import EvolutionaryBestClassifier
from algorithm.pipelinehelper import PipelineHelperClassifier, PipelineHelperTransformer
from config import CLASSIFICATION_TRACKS, AUTOML_PIPELINE, PARAM_GRID

from tracks.classification_tracks import anomaly_sine_track, random_rbf_track, agrawal_track, concept_drift_track, hyperplane_track, mixed_track, sea_track, sine_track, stagger_track
from utils import plot_track

if __name__ == '__main__':

    ensemble_model = tree.HoeffdingTreeClassifier()
    for track_name, track in tqdm(CLASSIFICATION_TRACKS):
        fig = plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=AUTOML_PIPELINE, param_grid=PARAM_GRID, sampling_rate=100),
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
            result_path=Path(f'./results/example_evoAutoMl'),
            verbose=2
        )