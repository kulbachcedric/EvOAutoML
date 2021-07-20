from river import metrics
from river.datasets import synth
from river.evaluate import Track
from river.metrics import Metrics, Accuracy, Rolling


def random_rbf_track(n_samples=10_000, seed=42):
    dataset = synth.RandomRBF(seed_model=7, seed_sample=seed).take(n_samples)
    track = Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def agrawal_track(n_samples=10_000, seed=42):
    dataset = synth.Agrawal(seed=seed).take(n_samples)
    track = Track("Agrawal + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def anomaly_sine_track(n_samples=10_000, seed=42):
    dataset = synth.AnomalySine(seed=42).take(n_samples)
    track = Track("Anomaly Sine + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def concept_drift_track(n_samples=10_000, seed=42):
    dataset = synth.ConceptDriftStream(seed=seed,
                                       stream=synth.Agrawal(classification_function=0),
                                       drift_stream=synth.Agrawal(classification_function=4),
                                       position = int(n_samples / 2),
                                       ).take(n_samples)
    #metric = Rolling(Accuracy(),window_size=1000)
    metric = Accuracy()
    track = Track("Agrawal Concept Drift + Accuracy", dataset, metric, n_samples)
    return track

def hyperplane_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed).take(n_samples)
    track = Track("Hyperplane + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def mixed_track(n_samples=10_000, seed=42):
    dataset = synth.Mixed(seed=seed).take(n_samples)
    track = Track("Mixed + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sea_track(n_samples=10_000, seed=42):
    dataset = synth.SEA(seed=seed).take(n_samples)
    track = Track("SEA + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sine_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    track = Track("SINE + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def stagger_track(n_samples=10_000, seed=42):
    dataset = synth.STAGGER(seed=seed).take(n_samples)
    track = Track("STAGGER + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track