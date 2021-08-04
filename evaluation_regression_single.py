from multiprocessing import Pool
from pathlib import Path

from river import tree, ensemble, linear_model, naive_bayes, neighbors
from tqdm import tqdm

from EvOAutoML.classification import EvolutionaryBestClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    POPULATION_SIZE, REGRESSION_TRACKS, N_SAMPLES, N_CHECKPOINTS
from EvOAutoML.regression import EvolutionaryBestRegressor

from EvOAutoML.utils import plot_track

def evaluate_single(track_dict):
    track = track_dict[1]
    plot_track(
        track=track,
        metric_name="R2",
        models={
            'EvoAutoML': EvolutionaryBestRegressor(population_size=POPULATION_SIZE, estimator=AUTOML_CLASSIFICATION_PIPELINE,
                                                    param_grid=CLASSIFICATION_PARAM_GRID, sampling_rate=250),
            # 'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
            ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
            'Hoeffding Tree': tree.HoeffdingTreeRegressor(),
            # 'FT',:tree.ExtremelyFastDecisionTreeClassifier(),
            'Linear Regression': linear_model.LinearRegression(),
            # ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
            #'GaussianNB': naive_bayes.GaussianNB(),
            # ('MNB', naive_bayes.MultinomialNB()),
            # ('PAC', linear_model.PAClassifier()),
            'KNN': neighbors.KNNRegressor(),
        },
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        result_path=Path(f'./results/regression/evaluation_single'),
        verbose=2)

if __name__ == '__main__':
    #output = [evaluate_single(t) for t in REGRESSION_TRACKS]
    pool = Pool(60)  # Create a multiprocessing Pool
    output = pool.map(evaluate_single, REGRESSION_TRACKS)
