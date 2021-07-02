from river import evaluate, metrics
from river.datasets import synth

def random_rbf_track(n_samples=10_000, seed=42):
    dataset = synth.RandomRBF(seed_model=7, seed_sample=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def agrawal_track(n_samples=10_000, seed=42):
    dataset = synth.Agrawal(seed=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def anomaly_sine_track(n_samples=10_000, seed=42):
    dataset = synth.AnomalySine(seed=42).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def concept_drift_track(n_samples=10_000, seed=42):
    dataset = synth.ConceptDriftStream(seed=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def hyperplane_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def mixed_track(n_samples=10_000, seed=42):
    dataset = synth.Mixed(seed=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sea_track(n_samples=10_000, seed=42):
    dataset = synth.SEA(seed=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sine_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def stagger_track(n_samples=10_000, seed=42):
    dataset = synth.STAGGER(seed=seed).take(n_samples)
    track = evaluate.Track("10K Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track