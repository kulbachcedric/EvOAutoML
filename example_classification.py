from pathlib import Path

import matplotlib.pyplot as plt
import river.naive_bayes
from river import tree, preprocessing, compose, feature_extraction, linear_model, ensemble, naive_bayes
from tqdm import tqdm

from algorithm.oaml import EvolutionaryBestClassifier
from algorithm.pipelinehelper import PipelineHelper, PipelineHelperTransformer, PipelineHelperClassifier
from tracks.classification_tracks import anomaly_sine_track, random_rbf_track, agrawal_track, concept_drift_track, \
    hyperplane_track, mixed_track, sea_track, sine_track, stagger_track

def plot_track(track, metric_name, models, n_samples, n_checkpoints, name=None):
    fig, ax = plt.subplots(figsize=(5, 5), nrows=3, dpi=300)
    for model_name, model in models.items():
        step = []
        error = []
        r_time = []
        memory = []
        for checkpoint in track(n_samples=n_samples, seed=42).run(model, n_checkpoints):
            step.append(checkpoint["Step"])
            error.append(checkpoint[metric_name])

            # Convert timedelta object into seconds
            r_time.append(checkpoint["Time"].total_seconds())
            # Make sure the memory measurements are in MB
            raw_memory, unit = float(checkpoint["Memory"][:-3]), checkpoint["Memory"][-2:]
            memory.append(raw_memory * 2**-10 if unit == 'KB' else raw_memory)
        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        ax[0].plot(step, error, label=model_name)
        ax[0].set_ylabel(metric_name)

        ax[1].plot(step, r_time, label=model_name)
        ax[1].set_ylabel('Time (seconds)')

        ax[2].plot(step, memory, label=model_name)
        ax[2].set_ylabel('Memory (MB)')
        ax[2].set_xlabel('Instances')

    plt.legend()
    plt.tight_layout()
    #plt.show()
    image_path = Path(f'results/image/')
    image_path.mkdir(parents=True, exist_ok=True)
    if name is not None:
        plt.savefig(str(image_path / f'{name}.pdf'))
    plt.clf()

    return fig



if __name__ == '__main__':
    tracks = [
        ('Random RBF', random_rbf_track),
        ('AGRAWAL', agrawal_track),
        ('Anomaly Sine', anomaly_sine_track),
        ('Concept Drift', concept_drift_track),
        ('Hyperplane', hyperplane_track),
        ('Mixed', mixed_track),
        ('SEA', sea_track),
        ('Sine', sine_track),
        ('STAGGER', stagger_track)
    ]
    estimator = compose.Pipeline(
        preprocessing.StandardScaler(),
        feature_extraction.PolynomialExtender(),
        tree.HoeffdingTreeClassifier()
    )

    automl_pipeline = compose.Pipeline(
        ('Scaler', PipelineHelperTransformer([
            #('StandardScaler', preprocessing.StandardScaler()),
            ('MinMaxScaler', preprocessing.MinMaxScaler()),
            ('MinAbsScaler', preprocessing.MaxAbsScaler()),
            ('RobustScaler', preprocessing.RobustScaler()),
            #('AdaptiveStandardScaler', preprocessing.AdaptiveStandardScaler()),
        ])),
         ('FeatureExtractor', PipelineHelperTransformer([
             #('Binarizer', preprocessing.Binarizer()),
             #('LDA', preprocessing.LDA()),
             ('PolynomialExtender', feature_extraction.PolynomialExtender()),
             ('RBF',feature_extraction.RBFSampler()),
         ])),
        ('Classifier', PipelineHelperClassifier([
            ('HT', tree.HoeffdingTreeClassifier()),
            #('FT', tree.ExtremelyFastDecisionTreeClassifier()),
            ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
            ('GNB', naive_bayes.GaussianNB())
        ]))
    )

    '''
    param_grid = {
        'PolynomialExtender__degree': [1,2,3],
        'PolynomialExtender__include_bias': [True, False],
        'HoeffdingTreeClassifier__split_criterion':  ['info_gain','gini','hellinger'],
        'HoeffdingTreeClassifier__tie_threshold':  [.01, .05, .1],
        'HoeffdingTreeClassifier__nb_threshold':  [0,.5],
        'HoeffdingTreeClassifier__binary_split':  [True,False],
        'HoeffdingTreeClassifier__max_size':  [50,100,150],
        'HoeffdingTreeClassifier__stop_mem_management':  [True, False],
        'HoeffdingTreeClassifier__remove_poor_attrs':  [True, False],
        'HoeffdingTreeClassifier__merit_preprune':  [True, False],
    }
    '''
    param_grid = {
        'Scaler': automl_pipeline.steps['Scaler'].generate({

        }),
        'FeatureExtractor' : automl_pipeline.steps['FeatureExtractor'].generate({

        }),
        'Classifier' : automl_pipeline.steps['Classifier'].generate({
            'HT__tie_threshold': [.01, .05, .1],
            'HT__max_size' : [10,100],
            #'FT__grace_period': [10,50,100]
        })
    }

    for track_name, track in tqdm(tracks):
        fig = plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                'EvoAutoML': EvolutionaryBestClassifier(estimator=automl_pipeline, param_grid=param_grid),
                'Unbounded HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
                'SRPC': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier()),
                'Bagging' : ensemble.BaggingClassifier(model=estimator),
                'Ada Boost' : ensemble.AdaBoostClassifier(model=estimator),
                'ARFC' : ensemble.AdaptiveRandomForestClassifier(),
                'LB' : ensemble.LeveragingBaggingClassifier(model=estimator),
                'Adwin Bagging' : ensemble.ADWINBaggingClassifier(model=estimator),
            },
            n_samples=10_000,
            n_checkpoints=100,
            name=track_name
        )