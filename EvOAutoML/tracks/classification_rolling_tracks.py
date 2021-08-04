from river import metrics
from river.datasets import synth, Elec2, ImageSegments
from river.evaluate import Track
from river.metrics import Rolling

METRIC_ROLLING_WINDOW = 1000


def random_rbf_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.RandomRBF(seed_model=7, seed_sample=seed, n_classes=10, n_features=50).take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Random RBF + Accuracy", dataset, metric, n_samples)
    return track


def led_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.LED(seed=seed, noise_percentage=.1)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("LED + Accuracy", dataset, metric, n_samples)
    return track


def agrawal_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Agrawal(seed=seed).take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Agrawal + Rolling Accuracy", dataset, metric, n_samples)
    return track


def anomaly_sine_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.AnomalySine(seed=seed, n_anomalies=(int(n_samples / 4))).take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Anomaly Sine + Rolling Accuracy", dataset, metric, n_samples)
    return track


def concept_drift_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.ConceptDriftStream(seed=seed,
                                       stream=synth.Agrawal(classification_function=0),
                                       drift_stream=synth.Agrawal(classification_function=4),
                                       position=int(n_samples / 2),
                                       ).take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Agrawal Concept Drift + Rolling Accuracy", dataset, metric, n_samples)
    return track


def hyperplane_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed, n_features=10, n_drift_features=5).take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Hyperplane + Rolling Accuracy", dataset, metric, n_samples)
    return track


def sea_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.SEA(seed=seed).take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("SEA + Rolling Accuracy", dataset, metric, n_samples)
    return track


def sine_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("SINE + Rolling Accuracy", dataset, metric, n_samples)
    return track


def elec2_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = Elec2().take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Elec2 + Rolling Accuracy", dataset, metric, n_samples)
    return track


def imagesegments_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = ImageSegments().take(n_samples)
    metric = Rolling(metric=metrics.Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("ImageSegmentation + Rolling Accuracy", dataset, metric, n_samples)
    return track
