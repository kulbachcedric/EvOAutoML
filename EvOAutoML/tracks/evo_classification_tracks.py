import datetime as dt
import time
import typing
from sklearn import datasets as sk_datasets
from river import metrics
from river import utils, stream
from river.base.typing import Stream
from river.datasets import synth, Elec2, ImageSegments
from river.evaluate import Track
from river.metrics import Accuracy

from EvOAutoML.classification import EvolutionaryBaggingClassifier


class EvoTrack(Track):

    def run(self, model, n_checkpoints=10):
        # Do the checkpoint logic
        step = self.n_samples // n_checkpoints
        checkpoints = range(0, self.n_samples, step)
        checkpoints = list(checkpoints)[1:] + [self.n_samples]

        population_size = model.population_size

        # A model might be used in multiple tracks. It's a sane idea to keep things pure and clone
        # the model so that there's no side effects.
        model = model.clone()

        yield from _progressive_evo_validation(
            dataset=self.dataset,
            model=model,
            metric=self.metric,
            population_metrics=[Accuracy() for _ in range(population_size)],
            checkpoints=iter(checkpoints),
            measure_time=True,
            measure_memory=True,
        )


def _progressive_evo_validation(
    dataset: Stream,
    model: EvolutionaryBaggingClassifier,
    population_metrics,
    metric: metrics.Metric,
    checkpoints: typing.Iterator[int],
    moment: typing.Union[str, typing.Callable] = None,
    delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
    measure_time=False,
    measure_memory=False,
):

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(
            f"{metric.__class__.__name__} metric is not compatible with {model}"
        )

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    pred_func = model.predict_one
    if utils.inspect.isclassifier(model) and not metric.requires_labels:
        pred_func = model.predict_proba_one

    preds = {}
    population_preds = {}
    for idx, metric in enumerate(population_metrics):
        population_preds[f'Individual {idx}'] = {}

    next_checkpoint = next(checkpoints, None)
    n_total_answers = 0
    if measure_time:
        start = time.perf_counter()

    for i, x, y in stream.simulate_qa(dataset, moment, delay, copy=True):

        # Question
        if y is None:
            preds[i] = pred_func(x=x)
            for idx, pred in enumerate(population_preds):
                population_preds[f'Individual {idx}'][i] = model[idx].predict_one(x)
            continue

        # Answer
        y_pred = preds.pop(i)
        y_population_pred = []
        for p in population_preds:
            t = population_preds[p].pop(i)
            y_population_pred.append(t)

        if y_pred != {} and y_pred is not None:
            metric.update(y_true=y, y_pred=y_pred)

        for idx, y_population_pred_i in enumerate(y_population_pred):
            if y_population_pred_i != {} and y_population_pred_i is not None:
                population_metrics[idx].update(y_true=y, y_pred=y_population_pred_i)

        model.learn_one(x=x, y=y)

        # Update the answer counter
        n_total_answers += 1
        if n_total_answers == next_checkpoint:
            if isinstance(metric, metrics.Metrics):
                results = {m.__class__.__name__: m.get() for m in metric}
            else:
                results = {metric.__class__.__name__: metric.get()}

            results["Accuracy"] = [results["Accuracy"]] * model.population_size
            results["Step"] = [n_total_answers] * model.population_size
            if measure_time:
                now = time.perf_counter()
                results["Time"] = [dt.timedelta(seconds=now - start)] * model.population_size
            if measure_memory:
                results["Memory"] = [model._memory_usage] * model.population_size
            results["Name"] = []
            results["Model Performance"] = []
            for idx, me in enumerate(population_metrics):
                results["Name"].append(f'Individual {idx}')
                results["Model Performance"].append(me.get())
                #results["Model Performance"].append(model.population_metrics[idx].get())

            yield results
            next_checkpoint = next(checkpoints, None)

def evo_rbf_accuracy_50_001_track(n_samples=10_000, seed=42):
    n_centroids = 50
    change_speed= .001
    dataset = synth.RandomRBFDrift(seed_model=7, seed_sample=seed,n_classes=5,n_features=50, n_centroids=n_centroids, change_speed=change_speed).take(n_samples)
    track = EvoTrack("RBF(50,0.001)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_rbf_accuracy_10_0001_track(n_samples=10_000, seed=42):
    n_centroids = 10
    change_speed= .0001
    dataset = synth.RandomRBFDrift(seed_model=7, seed_sample=seed,n_classes=5,n_features=50, n_centroids=n_centroids, change_speed=change_speed).take(n_samples)
    track = EvoTrack("RBF(10,0.0001)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_rbf_accuracy_10_001_track(n_samples=10_000, seed=42):
    n_centroids = 10
    change_speed= .001
    dataset = synth.RandomRBFDrift(seed_model=7, seed_sample=seed,n_classes=5,n_features=50, n_centroids=n_centroids, change_speed=change_speed).take(n_samples)
    track = EvoTrack("RBF(10,0.001)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_rbf_accuracy_50_0001_track(n_samples=10_000, seed=42):
    n_centroids = 50
    change_speed= .0001
    dataset = synth.RandomRBFDrift(seed_model=7, seed_sample=seed,n_classes=5,n_features=50, n_centroids=n_centroids, change_speed=change_speed).take(n_samples)
    track = EvoTrack("RBF(50,0.0001)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_sea_accuracy_50_track(n_samples=10_000, seed=42):
    width = 50
    dataset = synth.ConceptDriftStream(stream=synth.SEA(variant=1,seed=seed),
                                       drift_stream=synth.ConceptDriftStream(
                                           stream=synth.SEA(variant=0,seed=seed),
                                           drift_stream=synth.ConceptDriftStream(
                                               stream=synth.SEA(variant=2,seed=seed),
                                               drift_stream=synth.SEA(variant=3, seed=seed),
                                               seed=seed,
                                               position=int(n_samples*.75),
                                               width=width
                                           ),
                                           seed=seed,
                                           position=int(n_samples*.5),
                                           width=width
                                       ),
                                       seed=seed,
                                       position=int(n_samples*.25),
                                       width=width
                                       ).take(n_samples)

    track = EvoTrack("SEA(50)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_sea_accuracy_50000_track(n_samples=10_000, seed=42):
    width = 50_000
    dataset = synth.ConceptDriftStream(stream=synth.SEA(variant=1,seed=seed),
                                       drift_stream=synth.ConceptDriftStream(
                                           stream=synth.SEA(variant=0,seed=seed),
                                           drift_stream=synth.ConceptDriftStream(
                                               stream=synth.SEA(variant=2,seed=seed),
                                               drift_stream=synth.SEA(variant=3, seed=seed),
                                               seed=seed,
                                               position=int(n_samples*.75),
                                               width=width
                                           ),
                                           seed=seed,
                                           position=int(n_samples*.5),
                                           width=width
                                       ),
                                       seed=seed,
                                       position=int(n_samples*.25),
                                       width=width
                                       ).take(n_samples)
    track = EvoTrack("SEA(50,000)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_agrawal_accuracy_50_track(n_samples=10_000, seed=42):
    width = 50
    dataset = synth.ConceptDriftStream(stream=synth.Agrawal(classification_function=0,seed=seed),
                                       drift_stream=synth.ConceptDriftStream(
                                           stream=synth.Agrawal(classification_function=2,seed=seed),
                                           drift_stream=synth.ConceptDriftStream(
                                               stream=synth.Agrawal(classification_function=5,seed=seed),
                                               drift_stream=synth.Agrawal(classification_function=7, seed=seed),
                                               seed=seed,
                                               position=int(n_samples*.75),
                                               width=width
                                           ),
                                           seed=seed,
                                           position=int(n_samples*.5),
                                           width=width
                                       ),
                                       seed=seed,
                                       position=int(n_samples*.25),
                                       width=width
                                       ).take(n_samples)
    track = EvoTrack("Agrawal(50)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_agrawal_accuracy_50000_track(n_samples=10_000, seed=42):
    width = 50_000
    dataset = synth.ConceptDriftStream(stream=synth.Agrawal(classification_function=0,seed=seed),
                                       drift_stream=synth.ConceptDriftStream(
                                           stream=synth.Agrawal(classification_function=2,seed=seed),
                                           drift_stream=synth.ConceptDriftStream(
                                               stream=synth.Agrawal(classification_function=5,seed=seed),
                                               drift_stream=synth.Agrawal(classification_function=7, seed=seed),
                                               seed=seed,
                                               position=int(n_samples*.75),
                                               width=width
                                           ),
                                           seed=seed,
                                           position=int(n_samples*.5),
                                           width=width
                                       ),
                                       seed=seed,
                                       position=int(n_samples*.25),
                                       width=width
                                       ).take(n_samples)
    track = EvoTrack("Agrawal(50,000)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_led_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.LEDDrift(seed=seed, noise_percentage=.1, n_drift_features=4).take(n_samples)
    track = EvoTrack("LEDDrift()", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_hyperplane_accuracy_001_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed,n_features=50,n_drift_features=25,mag_change=.001).take(n_samples)
    track = EvoTrack("Hyperplane(50,0.001)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_hyperplane_accuracy_0001_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed,n_features=50,n_drift_features=25,mag_change=.0001).take(n_samples)
    track = EvoTrack("Hyperplane(50, 0.0001)", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_sine_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    track = EvoTrack("SINE()", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_elec2_accuracy_track(n_samples=10_000, seed=42):
    dataset = Elec2().take(n_samples)
    track = EvoTrack("Elec", dataset, metrics.Accuracy(), n_samples)
    return track

def evo_covtype_accuracy_track(n_samples=10_000, seed=42):
    dataset = stream.iter_sklearn_dataset(sk_datasets.fetch_covtype())
    track = EvoTrack('Covtype', dataset, metrics.Accuracy(), n_samples)
    return track

