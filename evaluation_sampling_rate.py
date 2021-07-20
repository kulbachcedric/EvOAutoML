from pathlib import Path

from river import tree, preprocessing, compose, naive_bayes, neighbors, ensemble, feature_extraction
from tqdm import tqdm

from algorithm.oaml import EvolutionaryBestClassifier
from algorithm.pipelinehelper import PipelineHelperClassifier, PipelineHelperTransformer
from config import CLASSIFICATION_TRACKS, automl_pipeline, param_grid, base_estimator

from tracks.classification_tracks import anomaly_sine_track, random_rbf_track, agrawal_track, concept_drift_track, hyperplane_track, mixed_track, sea_track, sine_track, stagger_track
from utils import plot_track

if __name__ == '__main__':

    sampling_rates = [1,5,10,25,50,100,150,200,250,500,750,1000]

    result_data = None
    for sampling_rate in  sampling_rates:
        for track_name, track in tqdm(CLASSIFICATION_TRACKS):
            data = plot_track(
                track=track,
                metric_name="Accuracy",
                models={
                    'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=automl_pipeline, param_grid=param_grid,sampling_rate=sampling_rate),
                    #'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
                    ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
                    'Bagging' : ensemble.BaggingClassifier(model=base_estimator),
                    'Ada Boost' : ensemble.AdaBoostClassifier(model=base_estimator),
                    'ARFC' : ensemble.AdaptiveRandomForestClassifier(),
                    'LB' : ensemble.LeveragingBaggingClassifier(model=base_estimator),
                    'Adwin Bagging' : ensemble.ADWINBaggingClassifier(model=base_estimator),
                },
                n_samples=10_000,
                n_checkpoints=1000,
                result_path=Path(f'./results/evaluation_sampling_rate/{track_name}_{sampling_rate}'),
                verbose=2
            )
            data['sampling_rate'] = len(data)*[sampling_rate]
            if result_data is None:
                result_data = data
            else:
                result_data = result_data.append(data)
    result_path = Path(f'./results/evaluation_sampling_rate.csv')
    result_path.mkdir(parents=True, exist_ok=True)
    result_data.to_csv(str(result_path))