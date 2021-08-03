
from river import datasets
from river.evaluate import Track
from river import metrics


def trump_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.TrumpApproval().take(n_samples)
    track = Track("TRUMP Approval + MSE", dataset, metrics.MSE(), n_samples)
    return track

def bikes_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.Bikes().take(n_samples)
    track = Track("BIKES + MSE", dataset, metrics.MSE(), n_samples)
    return track

def chickweights_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.ChickWeights().take(n_samples)
    track = Track("ChickWeights + MSE", dataset, metrics.MSE(), n_samples)
    return track

def movielens_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.MovieLens100K().take(n_samples)
    track = Track("MovieLens100K + MSE", dataset, metrics.MSE(), n_samples)
    return track

def restaurants_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.Restaurants().take(n_samples)
    track = Track("RESTAURANTS + MSE", dataset, metrics.MSE(), n_samples)
    return track

def taxis_mse_track(n_samples=10_000, seed=42):
    dataset = datasets.Taxis().take(n_samples)
    track = Track("TAXIS + MSE", dataset, metrics.MSE(), n_samples)
    return track
