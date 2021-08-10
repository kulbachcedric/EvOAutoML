
from river import metrics
from river.datasets import synth, Elec2, ImageSegments
from river.evaluate import Track
from river.metrics import Accuracy

from EvOAutoML.tracks.datasets import Covtype, PokerHand

def random_rbf_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.RandomRBF(seed_model=7, seed_sample=seed,n_classes=5,n_features=50, n_centroids=50).take(n_samples)
    track = Track("RBF", dataset, metrics.Accuracy(), n_samples)
    return track

def sea_accuracy_50_track(n_samples=10_000, seed=42):
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

    track = Track("SEA(50)", dataset, metrics.Accuracy(), n_samples)
    return track

def sea_accuracy_50000_track(n_samples=10_000, seed=42):
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
    track = Track("SEA(50,000)", dataset, metrics.Accuracy(), n_samples)
    return track

def agrawal_accuracy_50_track(n_samples=10_000, seed=42):
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
    track = Track("Agrawal(50)", dataset, metrics.Accuracy(), n_samples)
    return track

def agrawal_accuracy_50000_track(n_samples=10_000, seed=42):
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
    track = Track("Agrawal(50,000)", dataset, metrics.Accuracy(), n_samples)
    return track

def led_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.LED(seed=seed, noise_percentage=.1).take(n_samples)
    track = Track("LED()", dataset, metrics.Accuracy(), n_samples)
    return track

def hyperplane_accuracy_001_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed,n_features=50,n_drift_features=25,mag_change=.001).take(n_samples)
    track = Track("Hyperplane(50,0.001)", dataset, metrics.Accuracy(), n_samples)
    return track

def hyperplane_accuracy_0001_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed,n_features=50,n_drift_features=25,mag_change=.0001).take(n_samples)
    track = Track("Hyperplane(50, 0.0001)", dataset, metrics.Accuracy(), n_samples)
    return track

def sine_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    track = Track("SINE + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def elec2_accuracy_track(n_samples=10_000, seed=42):
    dataset = Elec2().take(n_samples)
    track = Track("Elec2 + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def covtype_accuracy_track(n_samples=10_000, seed=42):
    dataset = Covtype().take(n_samples)
    track = Track('Covtype + Accuracy', dataset, metrics.Accuracy(), n_samples)
    return track

