<p align="center">
  <img height="150px" src="docs/img/logo.png" alt="incremental dl logo">
</p>

<p align="center">
    EvO AutoML is a Python library for Evolution based Online AutoML.
    EvO AutoML ambition is to enable hyperparameter optimization for <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> pipelines build on <a href="https://riverml.xyz/latest/">river</a>.
</p>
<p align="center">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/EvoAutoML">
    <a href="https://codecov.io/gh/kulbachcedric/EvOAutoML" >
        <img src="https://codecov.io/gh/kulbachcedric/EvOAutoML/branch/master/graph/badge.svg?token=7RIEXKNR6K"/>
    </a>
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dw/EvOAutoML">
    <img alt="GitHub" src="https://img.shields.io/github/license/kulbachcedric/EvoAutoML"> 

</p>

# EvO AutoML

EvO AutoML is a Python library for Evolution based Online AutoML.

## 💈 Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install EvoAutoML.

```bash
pip install evoautoml
```

You can install the latest development version from GitHub as so:
```shell
pip install https://github.com/kulbachcedric/EvOAutoML//archive/refs/heads/master.zip
```
## 🍫 Quickstart
### Classification

```python

>>> from river import datasets, ensemble, evaluate, metrics, compose, optim
>>> from river import preprocessing, neighbors, naive_bayes, tree, linear_model
>>> from EvOAutoML import classification, pipelinehelper
>>> dataset = datasets.Phishing()
>>> model_pipeline = compose.Pipeline(
...     ('Scaler', pipelinehelper.PipelineHelperTransformer([
...         ('StandardScaler', preprocessing.StandardScaler()),
...         ('MinMaxScaler', preprocessing.MinMaxScaler()),
...         ('MinAbsScaler', preprocessing.MaxAbsScaler()),
...     ])),
...     ('Classifier', pipelinehelper.PipelineHelperClassifier([
...         ('HT', tree.HoeffdingTreeClassifier()),
...         ('LR', linear_model.LogisticRegression()),
...         ('GNB', naive_bayes.GaussianNB()),
...         ('KNN', neighbors.KNNClassifier()),
...     ])))
>>> model = classification.EvolutionaryBaggingClassifier(
...     model=model_pipeline,
...     param_grid={
...         'Scaler': model_pipeline.steps['Scaler'].generate({}),
...         'Classifier': model_pipeline.steps['Classifier'].generate({
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
>>> for x, y in dataset:
...     y_pred = model.predict_one(x)  # make a prediction
...     metric = metric.update(y, y_pred)  # update the metric
...     model = model.learn_one(x,y)  # make the model learn

```

## 📚 Cite
```
@inproceedings{DBLP:conf/pakdd/KulbachMBHB22,
  author    = {Cedric Kulbach and
               Jacob Montiel and
               Maroua Bahri and
               Marco Heyden and
               Albert Bifet},
  editor    = {Jo{\~{a}}o Gama and
               Tianrui Li and
               Yang Yu and
               Enhong Chen and
               Yu Zheng and
               Fei Teng},
  title     = {Evolution-Based Online Automated Machine Learning},
  booktitle = {Advances in Knowledge Discovery and Data Mining - 26th Pacific-Asia
               Conference, {PAKDD} 2022, Chengdu, China, May 16-19, 2022, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13280},
  pages     = {472--484},
  publisher = {Springer},
  year      = {2022},
  url       = {https://doi.org/10.1007/978-3-031-05933-9\_37},
  doi       = {10.1007/978-3-031-05933-9\_37},
  timestamp = {Tue, 17 May 2022 15:53:17 +0200},
  biburl    = {https://dblp.org/rec/conf/pakdd/KulbachMBHB22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## 🏫 Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>

