from river import compose, preprocessing, dummy, feature_extraction, tree, linear_model, neural_net, naive_bayes, \
    time_series, neighbors, optim
from river.neighbors import KNNClassifier

from EvOAutoML.pipelinehelper import PipelineHelperTransformer, PipelineHelperClassifier
from EvOAutoML.tracks.classification_tracks import random_rbf_accuracy_track, agrawal_accuracy_track, anomaly_sine_accuracy_track, \
    concept_drift_accuracy_track, \
    hyperplane_accuracy_track, mixed_accuracy_track, sea_accuracy_track, sine_accuracy_track, stagger_accuracy_track, elec2_accuracy_track, bananas_accuracy_track, creditcard_accuracy_track, \
    higgs_accuracy_track, imagesegments_accuracy_track, insects_accuracy_track, maliciousURL_accuracy_track, music_accuracy_track, pishing_accuracy_track, \
    smsspam_accuracy_track, trec07_accuracy_track

CLASSIFICATION_TRACKS = [
    ('Random RBF', random_rbf_accuracy_track),
    ('AGRAWAL', agrawal_accuracy_track),
    ('Anomaly Sine', anomaly_sine_accuracy_track),
    ('Concept Drift', concept_drift_accuracy_track),
    ('Hyperplane', hyperplane_accuracy_track),
    ('Mixed', mixed_accuracy_track),
    ('SEA', sea_accuracy_track),
    ('Sine', sine_accuracy_track),
    ('STAGGER', stagger_accuracy_track),
    ('ELEC2', elec2_accuracy_track),
    ('Bananas', bananas_accuracy_track),
    ('Credit Card', creditcard_accuracy_track),
    ('HIGGS', higgs_accuracy_track),
    ('Image Segments', imagesegments_accuracy_track),
    ('Insects', insects_accuracy_track),
    ('Malicious URL', maliciousURL_accuracy_track),
    ('Music', music_accuracy_track),
    ('Pishing', pishing_accuracy_track),
    ('SMS Spam', smsspam_accuracy_track),
    ('TREC', trec07_accuracy_track)
]

POPULATION_SIZE = 10

ENSEMBLE_CLASSIFIER = tree.HoeffdingTreeClassifier

AUTOML_CLASSIFICATION_PIPELINE = compose.Pipeline(
    ('Scaler', PipelineHelperTransformer([
        ('StandardScaler', preprocessing.StandardScaler()),
        ('MinMaxScaler', preprocessing.MinMaxScaler()),
        ('MinAbsScaler', preprocessing.MaxAbsScaler()),
        # todo create dummy
        #('RobustScaler', preprocessing.RobustScaler()),
        #('AdaptiveStandardScaler', preprocessing.AdaptiveStandardScaler()),
        #('LDA', preprocessing.LDA()),

    ])),
    #('FeatureExtractor', PipelineHelperTransformer([
    #    ('PolynomialExtender', feature_extraction.PolynomialExtender()),
        #('RBF', feature_extraction.RBFSampler()),
    #])),
    ('Classifier', PipelineHelperClassifier([
        ('HT', tree.HoeffdingTreeClassifier()),
        #('FT', tree.ExtremelyFastDecisionTreeClassifier()),
        ('LR', linear_model.LogisticRegression()),
        #('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
        ('GNB', naive_bayes.GaussianNB()),
        #('MNB', naive_bayes.MultinomialNB()),
        #('PAC', linear_model.PAClassifier()),
        ('KNN', neighbors.KNNClassifier()),
    ]))
)


CLASSIFICATION_PARAM_GRID = {
    #'Scaler': automl_pipeline.steps['Scaler'].generate({}),
    #'FeatureExtractor' : AUTOML_PIPELINE.steps['FeatureExtractor'].generate({
    #    'PolynomialExtender__degree' : [1,2],
    #    'PolynomialExtender__include_bias' : [True,False],
        #'RBF__n_components' : [2,10]
    #}),
    'Classifier' : AUTOML_CLASSIFICATION_PIPELINE.steps['Classifier'].generate({
        #'HT__tie_threshold': [.01, .05, .1],
        #'HT__max_size' : [10,50],
        #'HT__binary_split' : [True, False],
        'HT__max_depth' : [10,30,60,10,30,60],
        'HT__grace_period': [10, 100, 200,10, 100, 200],
        'HT__max_size': [5,10],
        #'LR__loss': [optim.losses.BinaryLoss,optim.losses.CrossEntropy],
        #'LR__l2': [.0,.01,.001],
        #'LR__optimizer': [optim.SGD,optim.Adam],
        'KNN__n_neighbors': [1,5,20],
        'KNN__window_size': [100,500,1000],
        'KNN__weighted': [True, False],
        'KNN__p': [1,2]

        #'HAT__tie_threshold': [.01, .05, .1],
        #'HAT__max_size' : [10,50],

        #'FT__max_depth': [10, 20, 50],
        #'FT__split_confidence': [1e-7],
        #'FT__tie_threshold': [0.05],
        #'FT__binary_split': [False],
        #'FT__max_size': [50, 100,200],
    })
}