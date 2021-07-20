from river import compose, preprocessing, feature_extraction, tree

from algorithm.pipelinehelper import PipelineHelperTransformer, PipelineHelperClassifier
from tracks.classification_tracks import random_rbf_track, agrawal_track, anomaly_sine_track, concept_drift_track, \
    hyperplane_track, mixed_track, sea_track, sine_track, stagger_track

CLASSIFICATION_TRACKS = [
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

base_estimator = compose.Pipeline(
    ('StandardScaler', preprocessing.StandardScaler()),
    ('PolynomialExtender', feature_extraction.PolynomialExtender()),
    ('clf', tree.HoeffdingTreeClassifier())
)

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
        #('KNN', neighbors.KNNClassifier()),
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
        #'KNN__n_neighbors' : [2,5,10],
        #'KNN__window_size' : [50,100,500]
    })
}