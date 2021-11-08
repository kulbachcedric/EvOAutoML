from river import compose, preprocessing, tree, linear_model, naive_bayes, neighbors, optim

from EvOAutoML.pipelinehelper import PipelineHelperRegressor
from EvOAutoML.pipelinehelper import PipelineHelperTransformer, PipelineHelperClassifier
from EvOAutoML.tracks.classification_tracks import *
from EvOAutoML.tracks.regression_tracks import *

POPULATION_SIZE = 10
N_SAMPLES = 1_000_000
N_CHECKPOINTS = 1000
SAMPLING_RATE = 1000

ENSEMBLE_CLASSIFIER = tree.HoeffdingTreeClassifier
ENSEMBLE_REGRESSOR = linear_model.LinearRegression

CLASSIFICATION_TRACKS = [
    ##('Poker Hand', pokerhand_accuracy_track),

    ('RBF(50,0.001)', rbf_accuracy_50_001_track),
    ('RBF(10,0.0001)', rbf_accuracy_10_0001_track),
    ('RBF(10,0.001)', rbf_accuracy_10_001_track),
    ('RBF(50,0.001)', rbf_accuracy_50_0001_track),

    ('SEA(50)', sea_accuracy_50_track),
    ('SEA(50000)', sea_accuracy_50000_track),

    ('Agrawal(50)', agrawal_accuracy_50_track),
    ('Agrawal(50000)', agrawal_accuracy_50000_track),

    ('LED()', led_accuracy_track),

    ('Hyperplane(50,0.001)', hyperplane_accuracy_001_track),
    ('Hyperplane(50,0.0001)', hyperplane_accuracy_0001_track),
    ('Sine', sine_accuracy_track),

    ('Covertype', covtype_accuracy_track),
    ('ELEC2', elec2_accuracy_track),
]

REGRESSION_TRACKS = [
    ('Trump Approval', trump_mse_track),
    # ('Bikes', bikes_mse_track),
    ('Chick Weights', chickweights_mse_track),
    # ('Movielens', movielens_mse_track),
    # ('Restaurants', restaurants_mse_track),
    # ('Taxi', taxis_mse_track)
]

AUTOML_CLASSIFICATION_PIPELINE = compose.Pipeline(
    ('Scaler', PipelineHelperTransformer([
        ('StandardScaler', preprocessing.StandardScaler()),
        ('MinMaxScaler', preprocessing.MinMaxScaler()),
        ('MinAbsScaler', preprocessing.MaxAbsScaler()),
        # todo create dummy
        # ('RobustScaler', preprocessing.RobustScaler()),
        # ('AdaptiveStandardScaler', preprocessing.AdaptiveStandardScaler()),
        # ('LDA', preprocessing.LDA()),

    ])),
    # ('FeatureExtractor', PipelineHelperTransformer([
    #    ('PolynomialExtender', feature_extraction.PolynomialExtender()),
    # ('RBF', feature_extraction.RBFSampler()),
    # ])),
    ('Classifier', PipelineHelperClassifier([
        ('HT', tree.HoeffdingTreeClassifier()),
        # ('FT', tree.ExtremelyFastDecisionTreeClassifier()),
        ('LR', linear_model.LogisticRegression()),
        # ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
        ('GNB', naive_bayes.GaussianNB()),
        # ('MNB', naive_bayes.MultinomialNB()),
        # ('PAC', linear_model.PAClassifier()),
        # ('ARF', ensemble.AdaptiveRandomForestClassifier()),
        ('KNN', neighbors.KNNClassifier()),
    ]))
)

CLASSIFICATION_PARAM_GRID = {
    'Scaler': AUTOML_CLASSIFICATION_PIPELINE.steps['Scaler'].generate({}),
    # 'FeatureExtractor' : AUTOML_PIPELINE.steps['FeatureExtractor'].generate({
    #    'PolynomialExtender__degree' : [1,2],
    #    'PolynomialExtender__include_bias' : [True,False],
    # 'RBF__n_components' : [2,10]
    # }),
    'Classifier': AUTOML_CLASSIFICATION_PIPELINE.steps['Classifier'].generate({
        'HT__max_depth': [10, 30, 60, 10, 30, 60],
        'HT__grace_period': [10, 100, 200, 10, 100, 200],
        'HT__max_size': [5, 10],
        #'LR__loss': [optim.losses.BinaryLoss,optim.losses.CrossEntropy],
        'LR__l2': [.0,.01,.001],
        #'LR__optimizer': [optim.SGD,optim.Adam],
        'KNN__n_neighbors': [1, 5, 20],
        'KNN__window_size': [100, 500, 1000],
        'KNN__weighted': [True, False],
        'KNN__p': [1, 2],
        # 'ARF__n_models': [5,10,5,10,5,10,5,10],
    })
}

AUTOML_REGRESSION_PIPELINE = compose.Pipeline(
    ('Scaler', PipelineHelperTransformer([
        ('StandardScaler', preprocessing.StandardScaler()),
        ('MinMaxScaler', preprocessing.MinMaxScaler()),
        ('MinAbsScaler', preprocessing.MaxAbsScaler()),
        # todo create dummy
        # ('RobustScaler', preprocessing.RobustScaler()),
        # ('AdaptiveStandardScaler', preprocessing.AdaptiveStandardScaler()),
        # ('LDA', preprocessing.LDA()),

    ])),
    ('Regressor', PipelineHelperRegressor([
        ('HT', tree.HoeffdingTreeRegressor()),
        # ('FT', tree.ExtremelyFastDecisionTreeClassifier()),
        # ('LR', linear_model.LinearRegression()),
        # ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
        # ('GNB', naive_bayes.GaussianNB()),
        # ('MNB', naive_bayes.MultinomialNB()),
        # ('PAC', linear_model.PAClassifier()),
        ('KNN', neighbors.KNNRegressor()),
    ]))
)

REGRESSION_PARAM_GRID = {
    # 'Scaler': automl_pipeline.steps['Scaler'].generate({}),
    # 'FeatureExtractor' : AUTOML_PIPELINE.steps['FeatureExtractor'].generate({
    #    'PolynomialExtender__degree' : [1,2],
    #    'PolynomialExtender__include_bias' : [True,False],
    # 'RBF__n_components' : [2,10]
    # }),
    'Regressor': AUTOML_REGRESSION_PIPELINE.steps['Regressor'].generate({
        # 'HT__tie_threshold': [.01, .05, .1],
        # 'HT__max_size' : [10,50],
        'HT__binary_split': [True, False],
        'HT__max_depth': [10, 30, 60, 10, 30, 60],
        'HT__grace_period': [10, 100, 200, 10, 100, 200],
        'HT__max_size': [5, 10],
        # 'LR__loss': [optim.losses.BinaryLoss,optim.losses.CrossEntropy],
        # 'LR__l2': [.0,.01,.001],
        # 'LR__optimizer': [optim.SGD,optim.Adam],
        'KNN__n_neighbors': [1, 5, 20],
        'KNN__window_size': [100, 500, 1000],
        # 'KNN__weighted': [True, False],
        'KNN__p': [1, 2]

        # 'HAT__tie_threshold': [.01, .05, .1],
        # 'HAT__max_size' : [10,50],

        # 'FT__max_depth': [10, 20, 50],
        # 'FT__split_confidence': [1e-7],
        # 'FT__tie_threshold': [0.05],
        # 'FT__binary_split': [False],
        # 'FT__max_size': [50, 100,200],
    })
}
