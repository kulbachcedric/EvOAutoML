
# EvO AutoML

EvO AutoML is a Python library for Evolution based Online AutoML.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install requirements.txt
```

## Usage
### Define Pipelines
```python
from EvOAutoML.pipeline import OnlinePipeline, OnlinePipelineHelper  
from EvOAutoML.transformer import ExtendedWindowedStandardScaler, ExtendedWindowedMinmaxScaler, ExtendedMissingValuesCleaner
from sklearn.linear_model import SGDClassifier
from skmultiflow.lazy import KNN, KNNAdwin
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.trees import HoeffdingTree, HoeffdingAdaptiveTreeClassifier  
from sklearn.naive_bayes import GaussianNB

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
```
### Define Search Space
```python
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
```
### Run Example
```python 
from EvOAutoML.oaml import EvolutionaryBestClassifier, BLASTClassifier, MetaStreamClassifier
from datasets.streams import StreamId, get_stream
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    stream = get_stream(streamId=StreamId.conceptdrift_sea.name, from_cache=True)
    model = EvolutionaryBestClassifier(population_size=5, estimator=pipe, param_grid=params,  
                                   window_size=window_size, metric = accuracy_score, sampling_rate=50)
    
    evaluator = EvaluatePrequential(show_plot=False,  
                                n_wait=100,  
                                batch_size=100,  
                                pretrain_size=200,  
                                max_samples=250,  
                                output_file='result.csv',  
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
```



