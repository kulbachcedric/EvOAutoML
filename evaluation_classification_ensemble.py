import itertools
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from river import ensemble, tree, linear_model, naive_bayes, neighbors

from EvOAutoML.classification import EvolutionaryBaggingClassifier, EvolutionaryLeveragingBaggingClassifer, \
    EvolutionaryOldestBaggingClassifier
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, \
    ENSEMBLE_CLASSIFIER, POPULATION_SIZE, N_SAMPLES, N_CHECKPOINTS, SAMPLING_RATE
from EvOAutoML.utils import plot_track, evaluate_track, evaluate_track_mlflow


RESULT_PATH = Path(f'./results/classification/evaluation_ensemble')

def evaluate_ensemble(track_tuple, model_tuple):
    track = track_tuple[1]
    output = evaluate_track_mlflow(
        track=track,
        metric_name="Accuracy",
        model_tuple=model_tuple,
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        verbose=2
    )
    eval_path = RESULT_PATH / f'ensemble_evaluation/{track_tuple[0]}'
    eval_path.mkdir(parents=True, exist_ok=True)
    output.to_csv(str(eval_path / f'{model_tuple[0]}.csv'))
    print(f'Finished Evaluating {model_tuple[0]} on {track_tuple[0]}')
    return output

ENSEMBLE_EVALUATION_MODELS = [
        ('EvoAutoML Bagging Oldest', EvolutionaryOldestBaggingClassifier(population_size=POPULATION_SIZE,
                                                                               model=AUTOML_CLASSIFICATION_PIPELINE,
                                                                               param_grid=CLASSIFICATION_PARAM_GRID,
                                                                               sampling_rate=SAMPLING_RATE)),
        ('EvoAutoML Bagging Best', EvolutionaryBaggingClassifier(population_size=POPULATION_SIZE, model=AUTOML_CLASSIFICATION_PIPELINE,
                                                           param_grid=CLASSIFICATION_PARAM_GRID, sampling_rate=SAMPLING_RATE)),
        ('ARF', ensemble.AdaptiveRandomForestClassifier()),
        ('Leveraging Bagging', ensemble.LeveragingBaggingClassifier(model=ENSEMBLE_CLASSIFIER())),
        ('Bagging' , ensemble.BaggingClassifier(model=ENSEMBLE_CLASSIFIER(),n_models=10)),
        ('SRPC', ensemble.SRPClassifier(n_models=10)),
        ('Hoeffding Tree', tree.HoeffdingTreeClassifier()),
        ('Logistic Regression', linear_model.LogisticRegression()),
        ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
        ('GaussianNB', naive_bayes.GaussianNB()),
        ('KNN', neighbors.KNNClassifier()),
    ]




if __name__ == '__main__':

    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    #output = evaluate_ensemble(CLASSIFICATION_TRACKS[1], ENSEMBLE_EVALUATION_MODELS[2])

    pool = Pool(60)  # Create a multiprocessing Pool
    output = pool.starmap(evaluate_ensemble, list(itertools.product(CLASSIFICATION_TRACKS, ENSEMBLE_EVALUATION_MODELS)))
    pool.close()
    pool.join()

    df = pd.concat(output)
    df.to_csv(str(RESULT_PATH / 'ensemble_evaluation.csv'))
