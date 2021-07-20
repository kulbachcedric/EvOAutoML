from pathlib import Path

from river import tree, preprocessing, compose, naive_bayes, neighbors, ensemble, feature_extraction, optim
from tqdm import tqdm

from algorithm.oaml import EvolutionaryBestClassifier
from algorithm.pipelinehelper import PipelineHelperClassifier, PipelineHelperTransformer
from config import CLASSIFICATION_TRACKS, base_estimator

from tracks.classification_tracks import anomaly_sine_track, random_rbf_track, agrawal_track, concept_drift_track, hyperplane_track, mixed_track, sea_track, sine_track, stagger_track
from river.facto import HOFMClassifier
from river.feature_extraction import PolynomialExtender
from river.linear_model import LinearRegression, PAClassifier, ALMAClassifier
from river.preprocessing import StandardScaler
from river.tree import ExtremelyFastDecisionTreeClassifier
from utils import plot_track

if __name__ == '__main__':


    automl_pipeline = compose.Pipeline(
        ('StandardScaler', StandardScaler()),
        ('PolynomialExtender', PolynomialExtender()),
        ('Clf', ExtremelyFastDecisionTreeClassifier())
    )


    param_grid = {
        'PolynomialExtender__degree': [1,2],
        'PolynomialExtender__include_bias' : [True,False],
        'Clf__grace_period': [10,100,200],
        'Clf__max_depth': [10,20,50],
        'Clf__min_samples_reevaluate': [20],
        'Clf__split_criterion': ['info_gain', 'gini', 'hellinger'],
        'Clf__split_confidence': [1e-7],
        'Clf__tie_threshold':  [0.05],
        'Clf__binary_split': [False],
        'Clf__max_size': [50,100],
        'Clf__memory_estimate_period':  [1000000],
        'Clf__stop_mem_management': [False,True],
        'Clf__remove_poor_attrs': [False,True],
        'Clf__merit_preprune': [False,True]
    }
    # '''

    ensemble_model = tree.HoeffdingTreeClassifier()
    for track_name, track in tqdm(CLASSIFICATION_TRACKS):
        fig = plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=automl_pipeline, param_grid=param_grid,sampling_rate=100),
                'Pipeline': base_estimator,
                ##'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),n_models=10),
                #'Bagging' : ensemble.BaggingClassifier(model=ensemble_model),
                #'Ada Boost' : ensemble.AdaBoostClassifier(model=ensemble_model),
                #'ARFC' : ensemble.AdaptiveRandomForestClassifier(),
                #'LB' : ensemble.LeveragingBaggingClassifier(model=ensemble_model),
                #'Adwin Bagging' : ensemble.ADWINBaggingClassifier(model=ensemble_model),
            },
            n_samples=10_000,
            n_checkpoints=1000,
            result_path=Path(f'./results/evaluation_simple'),
            verbose=2
        )