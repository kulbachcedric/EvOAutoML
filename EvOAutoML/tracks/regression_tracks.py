
from river import datasets
from river import metrics
from river.evaluate import Track


def trump_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.TrumpApproval().take(n_samples)
    track = Track("TRUMP Approval + R2", dataset, metrics.R2(), n_samples)
    return track

def chickweights_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.ChickWeights().take(n_samples)
    track = Track("ChickWeights + R2", dataset, metrics.R2(), n_samples)
    return track
