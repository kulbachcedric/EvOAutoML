from river import (
    compose,
    linear_model,
    naive_bayes,
    neighbors,
    preprocessing,
    tree,
)

from EvOAutoML.pipelinehelper import (
    PipelineHelperClassifier,
    PipelineHelperRegressor,
    PipelineHelperTransformer,
)

AUTOML_CLASSIFICATION_PIPELINE = compose.Pipeline(
    (
        "Scaler",
        PipelineHelperTransformer(
            [
                ("StandardScaler", preprocessing.StandardScaler()),
                ("MinMaxScaler", preprocessing.MinMaxScaler()),
                ("MinAbsScaler", preprocessing.MaxAbsScaler()),
            ]
        ),
    ),
    (
        "Classifier",
        PipelineHelperClassifier(
            [
                ("HT", tree.HoeffdingTreeClassifier()),
                # ('FT', tree.ExtremelyFastDecisionTreeClassifier()),
                ("LR", linear_model.LogisticRegression()),
                # ('HAT', tree.HoeffdingAdaptiveTreeClassifier()),
                ("GNB", naive_bayes.GaussianNB()),
                # ('MNB', naive_bayes.MultinomialNB()),
                # ('PAC', linear_model.PAClassifier()),
                # ('ARF', ensemble.AdaptiveRandomForestClassifier()),
                ("KNN", neighbors.KNNClassifier()),
            ]
        ),
    ),
)

CLASSIFICATION_PARAM_GRID = {
    "Scaler": AUTOML_CLASSIFICATION_PIPELINE.steps["Scaler"].generate({}),
    "Classifier": AUTOML_CLASSIFICATION_PIPELINE.steps["Classifier"].generate(
        {
            "HT__max_depth": [10, 30, 60, 10, 30, 60],
            "HT__grace_period": [10, 100, 200, 10, 100, 200],
            "HT__max_size": [5, 10],
            "LR__l2": [0.0, 0.01, 0.001],
            "KNN__n_neighbors": [1, 5, 20],
            "KNN__window_size": [100, 500, 1000],
            "KNN__weighted": [True, False],
            "KNN__p": [1, 2],
        }
    ),
}

AUTOML_REGRESSION_PIPELINE = compose.Pipeline(
    (
        "Scaler",
        PipelineHelperTransformer(
            [
                ("StandardScaler", preprocessing.StandardScaler()),
                ("MinMaxScaler", preprocessing.MinMaxScaler()),
                ("MinAbsScaler", preprocessing.MaxAbsScaler()),
            ]
        ),
    ),
    (
        "Regressor",
        PipelineHelperRegressor(
            [
                ("HT", tree.HoeffdingTreeRegressor()),
                ("KNN", neighbors.KNNRegressor()),
            ]
        ),
    ),
)

REGRESSION_PARAM_GRID = {
    "Regressor": AUTOML_REGRESSION_PIPELINE.steps["Regressor"].generate(
        {
            "HT__binary_split": [True, False],
            "HT__max_depth": [10, 30, 60, 10, 30, 60],
            "HT__grace_period": [10, 100, 200, 10, 100, 200],
            "HT__max_size": [5, 10],
            "KNN__n_neighbors": [1, 5, 20],
            "KNN__window_size": [100, 500, 1000],
            "KNN__p": [1, 2],
        }
    )
}
