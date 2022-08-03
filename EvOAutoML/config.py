from river import compose, preprocessing, tree, linear_model, naive_bayes, neighbors, optim

from EvOAutoML.pipelinehelper import PipelineHelperRegressor
from EvOAutoML.pipelinehelper import PipelineHelperTransformer, PipelineHelperClassifier

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
