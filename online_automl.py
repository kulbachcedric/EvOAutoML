import os
from pathlib import Path

from pipelinehelper import PipelineHelper
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from skmultiflow.evaluation import EvaluatePrequential
import argparse
from automlstreams.meta import MetaClassifier, LastBestClassifier
from skmultiflow.lazy import KNN, KNNAdwin, SAMKNN
from skmultiflow.rules import VeryFastDecisionRulesClassifier
from skmultiflow.meta import LeverageBagging, OzaBaggingAdwin
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.trees import HoeffdingTree, HAT, HoeffdingAdaptiveTreeClassifier
from skmultiflow.trees import HoeffdingTree as HT
import numpy as np

from EvOAutoML.oaml import EvolutionaryBestClassifier, BLASTClassifier, MetaStreamClassifier
from EvOAutoML.pipeline import OnlinePipeline, OnlinePipelineHelper
from EvOAutoML.transformer import ExtendedWindowedStandardScaler, ExtendedWindowedMinmaxScaler, ExtendedMissingValuesCleaner
from datasets.streams import StreamId, get_stream

MAX_SAMPLES = 20000

BATCH_SIZE = 50


def run(window_size, model, topic):
    print(f'Running demo for file=/_datasets/{topic}.csv')
    # stream = FileStream(f'/_datasets/{topic}.csv')
    stream = get_stream(streamId=topic, from_cache=True)
    stream.prepare_for_use()
    model_name = model.__class__.__name__
    if not Path(f'./results/experiment_{window_size}').exists():
        os.mkdir(f'./results/experiment_{window_size}')
    evaluator = EvaluatePrequential(show_plot=False,
                                    n_wait=100,
                                    batch_size=100,
                                    pretrain_size=200,
                                    max_samples=MAX_SAMPLES,
                                    output_file=f'./results/experiment_{window_size}/online.{model_name}.{topic}.csv',
                                    metrics=['accuracy',
                                             'model_size',
                                             'running_time',
                                             'kappa',
                                             'kappa_t',
                                             'kappa_m',
                                             'true_vs_predicted',
                                             'precision',
                                             'recall',
                                             'f1'])
    evaluator.evaluate(stream=stream, model=model)
    # model.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--window_sizes',
                        default=[1,25, 50, 100, 250, 500, 1000, 2000],
                        type=int,
                        nargs='+',
                        help='Which window sizes should be processed')
    args = parser.parse_args()
    window_sizes = args.window_sizes

    topics = [
        ###StreamId.agrawal_gen.name,
        ###StreamId.stagger_gen.name,
        StreamId.hyperplane_gen.name,
        StreamId.led_gen.name,
        StreamId.rbf_gen.name,
        ###StreamId.sea_gen.name,
        StreamId.covtype.name,
        StreamId.elec.name,
        StreamId.pokerhand.name,
        StreamId.conceptdrift_sea.name,
        StreamId.conceptdrift_agrawal.name,
        StreamId.conceptdrift_stagger.name
    ]
    pipe = OnlinePipeline([
        ('trans', OnlinePipelineHelper([
            ('mvc', ExtendedMissingValuesCleaner()),
            ('wmms', ExtendedWindowedMinmaxScaler()),
            ('wss', ExtendedWindowedStandardScaler())
        ])),
        ('clf', OnlinePipelineHelper([
            ('gnb' , GaussianNB()),
            ('sgd', SGDClassifier()),
            ('hat', HoeffdingTree()),
            ('ahat', HoeffdingAdaptiveTreeClassifier()),
            ('knna', KNNAdwin()),
            ('knn', KNN()),
            ('mlp', PerceptronMask()),
        ]))
    ])

    params = {
        'trans__selected_model': pipe.named_steps['trans'].generate({
            'mvc__strategy' : ['zero', 'mean', 'median'],
        }),
        'clf__selected_model': pipe.named_steps['clf'].generate({
            'knna__n_neighbors': [2, 5, 10],
            'knna__leaf_size': [10, 20, 30, 40, 50],

            'knn__n_neighbors': [2, 5, 10],
            'knn__leaf_size': [10, 20, 30, 40, 50],

            'hat__tie_threshold': [0.01,0.05, 0.1],
            'hat__split_criterion': ['gini','info_gain','hellinger'],
            'hat__binary_split': [True,False],
            'hat__remove_poor_atts' : [True,False],

            'ahat__tie_threshold': [0.01, 0.05, 0.1],
            'ahat__split_criterion': ['gini', 'info_gain', 'hellinger'],
            'ahat__binary_split': [True, False],
            'ahat__remove_poor_atts': [True, False],

            'sgd__learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
            'sgd__shuffle' : [True, False],
            'sgd__eta0': [1.0],


        }),
    }

    models = [
        'evolutionary_best',
        'meta',
        'last_best',
        'hoeffding_tree',
        'knn',
        'perceptron_mask',
        'sgd',
        'hat',
        'leverage_bagging',
        'oza_bagging_adwin'
    ]
    print([m.__class__.__name__ for m in models])

    for window_size in window_sizes:
        print(f'Process window size {window_size}')
        for topic in topics:
            for model in models:
                if model == 'evolutionary_best':
                    model = EvolutionaryBestClassifier(population_size=5, estimator=pipe, param_grid=params,
                                                       window_size=window_size, metric=accuracy_score)
                elif model == 'meta':
                    model = MetaStreamClassifier(meta_estimator=SGDClassifier(),
                                           base_estimators=[KNNAdwin(), HT()],
                                           mfe_groups=[
                                               'general',
                                               # 'statistical',
                                               'info-theory'
                                           ],
                                           window_size=window_size,
                                           active_learning=True)
                elif model == 'last_best':
                    model = BLASTClassifier(estimators=[KNNAdwin(), HT()], window_size=window_size)
                elif model == 'hoeffding_tree':
                    model = HoeffdingTree()
                elif model == 'knn':
                    model = KNN(max_window_size=window_size)
                elif model == 'perceptron_mask':
                    model = PerceptronMask()
                elif model == 'sgd':
                    model = SGDClassifier()
                elif model == 'hat':
                    model = HAT()
                elif model == 'leverage_bagging':
                    model = LeverageBagging()
                elif model == 'oza_bagging_adwin':
                    model = OzaBaggingAdwin()
                print('\n', model.__class__.__name__, topic, ':\n')
                run(window_size=window_size, model=model, topic=topic)
