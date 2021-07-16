from pathlib import Path

from river import tree, preprocessing, compose, naive_bayes, neighbors, ensemble, feature_extraction
from tqdm import tqdm

from algorithm.oaml import EvolutionaryBestClassifier
from algorithm.pipelinehelper import PipelineHelperClassifier, PipelineHelperTransformer

from tracks.classification_tracks import anomaly_sine_track, random_rbf_track, agrawal_track, concept_drift_track, hyperplane_track, mixed_track, sea_track, sine_track, stagger_track
from utils import plot_track

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
    sampling_rates = [1,5,10,25,50,100,150,200,250,500,750,1000]


    automl_pipeline = compose.Pipeline(
        ('Scaler', PipelineHelperTransformer([
            ('StandardScaler', preprocessing.StandardScaler()),
            ('MinMaxScaler', preprocessing.MinMaxScaler()),
            ('MinAbsScaler', preprocessing.MaxAbsScaler()),
            #('RobustScaler', preprocessing.RobustScaler()),
            #('AdaptiveStandardScaler', preprocessing.AdaptiveStandardScaler()),
            #('LDA', preprocessing.LDA()),

        ])),
        ('FeatureExtractor', PipelineHelperTransformer([
            ('PolynomialExtender', feature_extraction.PolynomialExtender()),
            #('RBF', feature_extraction.RBFSampler()),
        ])),
        ('Classifier', PipelineHelperClassifier([
            ('HT', tree.HoeffdingTreeClassifier()),
            ('FT', tree.ExtremelyFastDecisionTreeClassifier()),
            ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
            #('GNB', naive_bayes.GaussianNB()),
            #('MNB', naive_bayes.MultinomialNB()),
            #('PAC', linear_model.PAClassifier()),
            ('KNN', neighbors.KNNClassifier()),
        ]))
    )


    param_grid = {
        #'Scaler': automl_pipeline.steps['Scaler'].generate({}),
        'FeatureExtractor' : automl_pipeline.steps['FeatureExtractor'].generate({
            'PolynomialExtender__degree' : [1,2],
            #'RBF__n_components' : [2,10]
        }),
        'Classifier' : automl_pipeline.steps['Classifier'].generate({
            'HT__tie_threshold': [.01, .05, .1],
            'HT__max_size' : [10,50],
            'HAT__tie_threshold': [.01, .05, .1],
            'HAT__max_size' : [10,50],
            'KNN__n_neighbors' : [2,5,10],
            'KNN__window_size' : [50,100,500]
        })
    }
    # '''
    result_data = None
    ensemble_model = tree.HoeffdingTreeClassifier()
    for sampling_rate in  sampling_rates:
        for track_name, track in tqdm(tracks):
            data = plot_track(
                track=track,
                metric_name="Accuracy",
                models={
                    'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=automl_pipeline, param_grid=param_grid,sampling_rate=sampling_rate),
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
                result_path=Path(f'./results/evaluation_sampling_rate/{track_name}_{sampling_rate}'),
                verbose=2
            )
            data['sampling_rate'] = [len(data)*sampling_rate]
            if result_data is None:
                result_data = data
            else:
                result_data.append(data)
    result_path = Path(f'./results/evaluation_sampling_rate.csv')
    result_path.mkdir(parents=True, exist_ok=True)
    result_data.to_csv(str(result_path))