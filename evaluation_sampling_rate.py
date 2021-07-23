import itertools
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from river import ensemble
from tqdm import tqdm

from EvOAutoML.oaml import EvolutionaryBestClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, BASE_CLASSIFIER, ENSEMBLE_CLASSIFIER
import pandas as pd
from EvOAutoML.utils import plot_track

def evaluate_sampling_rate(sampling_rate:int,track_tuple:Tuple):
    track_name = track_tuple[0]
    track = track_tuple[1]
    data = plot_track(
        track=track,
        metric_name="Accuracy",
        models={
            'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=AUTOML_CLASSIFICATION_PIPELINE, param_grid=CLASSIFICATION_PARAM_GRID, sampling_rate=sampling_rate),
            #'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
            ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
            'Bagging' : ensemble.BaggingClassifier(model=ENSEMBLE_CLASSIFIER),
            'Ada Boost' : ensemble.AdaBoostClassifier(model=ENSEMBLE_CLASSIFIER),
            'ARFC' : ensemble.AdaptiveRandomForestClassifier(),
            'LB' : ensemble.LeveragingBaggingClassifier(model=ENSEMBLE_CLASSIFIER),
            'Adwin Bagging' : ensemble.ADWINBaggingClassifier(model=ENSEMBLE_CLASSIFIER),
        },
        n_samples=10_000,
        n_checkpoints=1000,
        result_path=Path(f'./results/evaluation_sampling_rate/{track_name}_{sampling_rate}'),
        verbose=2
    )
    data['sampling_rate'] = len(data)*[sampling_rate]
    data['track'] = len(data)*[track_name]
    return data


if __name__ == '__main__':
    sampling_rates = [1,5,10,25,50,100,150,200,250,500,750,1000]

    testing_configurations = list(itertools.product(sampling_rates,CLASSIFICATION_TRACKS))

    pool = Pool(60)  # Create a multiprocessing Pool
    output = pool.starmap(evaluate_sampling_rate, testing_configurations)
    result_data = pd.concat(output)

    result_path = Path(f'./results')
    result_path.mkdir(parents=True, exist_ok=True)
    result_path = result_path / 'evaluation_sampling_rate.xlsx'
    result_data.to_excel(str(result_path))