
from river import metrics
from river.datasets import synth, Elec2, ImageSegments
from river.evaluate import Track
from river.metrics import Accuracy

def random_rbf_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.RandomRBF(seed_model=7, seed_sample=seed,n_classes=5,n_features=50, n_centroids=50).take(n_samples)
    track = Track("Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def led_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.LED(seed=seed, noise_percentage=.1)
    track = Track("LED + Accuracy", dataset, metrics.Accuracy, n_samples)
    return track

def agrawal_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Agrawal(seed=seed).take(n_samples)
    track = Track("Agrawal + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def anomaly_sine_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.AnomalySine(seed=seed,n_anomalies=max(int(n_samples/4),10_000)).take(n_samples)
    track = Track("Anomaly Sine + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def concept_drift_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.ConceptDriftStream(seed=seed,
                                       stream=synth.Agrawal(classification_function=0),
                                       drift_stream=synth.Agrawal(classification_function=4),
                                       position = int(n_samples / 2),
                                       ).take(n_samples)
    #metric = Rolling(Accuracy(),window_size=1000)
    metric = Accuracy()
    track = Track("Agrawal Concept Drift + Accuracy", dataset, metric, n_samples)
    return track

def hyperplane_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed,n_features=10,n_drift_features=5,mag_change=.001).take(n_samples)
    track = Track("Hyperplane + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sea_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.SEA(seed=seed).take(n_samples)
    track = Track("SEA + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sine_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    track = Track("SINE + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def elec2_accuracy_track(n_samples=10_000, seed=42):
    dataset = Elec2().take(n_samples)
    track = Track("Elec2 + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

