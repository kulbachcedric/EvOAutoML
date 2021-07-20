from pathlib import Path

from river import ensemble
from tqdm import tqdm

from EvOAutoML.oaml import EvolutionaryBestClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_PIPELINE, PARAM_GRID, BASE_ESTIMATOR

from EvOAutoML.utils import plot_track

if __name__ == '__main__':

    sampling_rates = [1,5,10,25,50,100,150,200,250,500,750,1000]

    result_data = None
    for sampling_rate in  sampling_rates:
        for track_name, track in tqdm(CLASSIFICATION_TRACKS):
            data = plot_track(
                track=track,
                metric_name="Accuracy",
                models={
                    'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=AUTOML_PIPELINE, param_grid=PARAM_GRID, sampling_rate=sampling_rate),
                    #'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
                    ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
                    'Bagging' : ensemble.BaggingClassifier(model=BASE_ESTIMATOR),
                    'Ada Boost' : ensemble.AdaBoostClassifier(model=BASE_ESTIMATOR),
                    'ARFC' : ensemble.AdaptiveRandomForestClassifier(),
                    'LB' : ensemble.LeveragingBaggingClassifier(model=BASE_ESTIMATOR),
                    'Adwin Bagging' : ensemble.ADWINBaggingClassifier(model=BASE_ESTIMATOR),
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