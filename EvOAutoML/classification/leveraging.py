import collections
import typing

from river import base, metrics

from EvOAutoML.base.evolution import EvolutionaryLeveragingBaggingEstimator
from EvOAutoML.config import AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID

class EvolutionaryLeveragingBaggingClassifer(EvolutionaryLeveragingBaggingEstimator, base.Classifier):
    """
    Evolutionary Leveraging Bagging Classifier follows a Leveraging Bagging approach to update the population of
    estimator pipelines.

    Parameters
    ----------
    model
        A river model or model pipeline that can be configured
        by the parameter grid.
    param_grid
        A parameter grid, that represents the configuration space of the model.
    population_size
        The population size estimates the size of the population as
        well as the size of the ensemble used for the prediction.
    sampling_size
        The sampling size estimates how many models are mutated
        within one mutation step.
    metric
        The river metric that should be optimised.
    sampling_rate
        The sampling rate estimates the number of samples that are executed
        before a mutation step takes place.
    w
        Indicates the average number of events. This is the lambda parameter
        of the Poisson distribution used to compute the re-sampling weight.
    adwin_delta
        The delta parameter for the ADWIN change detector.
    bagging_method
        The bagging method to use. Can be one of the following:<br/>
            * 'bag' - Leveraging Bagging using ADWIN.<br/>
            * 'me' - Assigns $weight=1$ if sample is misclassified,
              otherwise $weight=error/(1-error)$.<br/>
            * 'half' - Use resampling without replacement for half
            of the instances.<br/>
            * 'wt' - Resample without taking out all instances.<br/>
            * 'subag' - Resampling without replacement.<br/>
    seed
        Random number generator seed for reproducibility.

    Examples
    --------
    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing
    >>> dataset = datasets.Phishing()
    >>> model = ensemble.EvolutionaryLeveragingBaggingClassifer(
    ...     model=compose.Pipeline(
    ...         ('Scaler', PipelineHelperTransformer([
    ...             ('StandardScaler', preprocessing.StandardScaler()),
    ...             ('MinMaxScaler', preprocessing.MinMaxScaler()),
    ...             ('MinAbsScaler', preprocessing.MaxAbsScaler()),
    ...         ])),
    ...         ('Classifier', PipelineHelperClassifier([
    ...             ('HT', tree.HoeffdingTreeClassifier()),
    ...             ('LR', linear_model.LogisticRegression()),
    ...             ('GNB', naive_bayes.GaussianNB()),
    ...             ('KNN', neighbors.KNNClassifier()),
    ...         ]))
    ...     ),
    ...     param_grid={
    ...         'Scaler': AUTOML_CLASSIFICATION_PIPELINE.steps['Scaler'].generate({}),
    ...         'Classifier': AUTOML_CLASSIFICATION_PIPELINE.steps['Classifier'].generate({
    ...             'HT__max_depth': [10, 30, 60, 10, 30, 60],
    ...             'HT__grace_period': [10, 100, 200, 10, 100, 200],
    ...             'HT__max_size': [5, 10],
    ...             'LR__l2': [.0,.01,.001],
    ...             'KNN__n_neighbors': [1, 5, 20],
    ...             'KNN__window_size': [100, 500, 1000],
    ...             'KNN__weighted': [True, False],
    ...             'KNN__p': [1, 2],
    ...         })
    ...     },
    ...     seed=42
    ... )
    >>> metric = metrics.F1()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 88.73%
    """
    def __init__(
            self,
            model: base.Classifier = AUTOML_CLASSIFICATION_PIPELINE,
            param_grid=CLASSIFICATION_PARAM_GRID,
            population_size:int=10,
            sampling_size:int=1,
            metric=metrics.Accuracy,
            sampling_rate:int =1000,
            w:int = 6,
            adwin_delta: float = 0.002,
            bagging_method: str = "bag",
            seed: int = 42,
    ):

        super().__init__(
            model=model,
            param_grid=param_grid,
            population_size=population_size,
            sampling_size=sampling_size,
            metric=metric,
            sampling_rate=sampling_rate,
            w=w,
            adwin_delta=adwin_delta,
            bagging_method=bagging_method,
            seed=seed
        )
    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred