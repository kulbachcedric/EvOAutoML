
from river import metrics
from river.datasets import synth, Bananas, Elec2, CreditCard, Higgs, ImageSegments, Insects, MaliciousURL, Music, Phishing, SMSSpam, TREC07
from river.evaluate import Track
from river.metrics import Accuracy, Rolling

METRIC_ROLLING_WINDOW = 250

def random_rbf_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.RandomRBF(seed_model=7, seed_sample=seed,n_classes=10,n_features=200).take(n_samples)
    track = Track("Random RBF + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def agrawal_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Agrawal(seed=seed).take(n_samples)
    track = Track("Agrawal + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def anomaly_sine_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.AnomalySine(seed=42).take(n_samples)
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
    dataset = synth.Hyperplane(seed=seed,n_features=100).take(n_samples)
    track = Track("Hyperplane + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def mixed_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Mixed(seed=seed).take(n_samples)
    track = Track("Mixed + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sea_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.SEA(seed=seed).take(n_samples)
    track = Track("SEA + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def sine_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    track = Track("SINE + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def stagger_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.STAGGER(seed=seed).take(n_samples)
    track = Track("STAGGER + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def elec2_accuracy_track(n_samples=10_000, seed=42):
    dataset = Elec2().take(n_samples)
    track = Track("Elec2 + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def bananas_accuracy_track(n_samples=10_000, seed=42):
    dataset = Bananas().take(n_samples)
    track = Track("Bananas + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def creditcard_accuracy_track(n_samples=10_000, seed=42):
    dataset = CreditCard().take(n_samples)
    track = Track("CreditCard + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def higgs_accuracy_track(n_samples=10_000, seed=42):
    dataset = Higgs().take(n_samples)
    track = Track("Higgs + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def imagesegments_accuracy_track(n_samples=10_000, seed=42):
    dataset = ImageSegments().take(n_samples)
    track = Track("ImageSegmentation + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def insects_accuracy_track(n_samples=10_000, seed=42):
    dataset = Insects().take(n_samples)
    track = Track("Insects + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def maliciousURL_accuracy_track(n_samples=10_000, seed=42):
    dataset = MaliciousURL().take(n_samples)
    track = Track("MaliciousURL + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def music_accuracy_track(n_samples=10_000, seed=42):
    dataset = Music().take(n_samples)
    track = Track("Music + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def pishing_accuracy_track(n_samples=10_000, seed=42):
    dataset = Phishing().take(n_samples)
    track = Track("Phishing + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def smsspam_accuracy_track(n_samples=10_000, seed=42):
    dataset = SMSSpam().take(n_samples)
    track = Track("SMSSpam + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def trec07_accuracy_track(n_samples=10_000, seed=42):
    dataset = TREC07().take(n_samples)
    track = Track("TREC07 + Accuracy", dataset, metrics.Accuracy(), n_samples)
    return track

def random_rbf_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.RandomRBF(seed_model=7, seed_sample=seed,n_classes=10,n_features=200).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Random RBF + Rolling Accuracy", dataset, metric, n_samples)
    return track

def agrawal_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Agrawal(seed=seed).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Agrawal + Rolling Accuracy", dataset, metric, n_samples)
    return track

def anomaly_sine_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.AnomalySine(seed=42).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Anomaly Sine + Rolling Accuracy", dataset, metric, n_samples)
    return track

def concept_drift_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.ConceptDriftStream(seed=seed,
                                       stream=synth.Agrawal(classification_function=0),
                                       drift_stream=synth.Agrawal(classification_function=4),
                                       position = int(n_samples / 2),
                                       ).take(n_samples)
    metric = Rolling(Accuracy(),window_size=METRIC_ROLLING_WINDOW)
    track = Track("Agrawal Concept Drift + Rolling Accuracy", dataset, metric, n_samples)
    return track

def hyperplane_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Hyperplane(seed=seed,n_features=100).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Hyperplane + Rolling Accuracy", dataset, metric, n_samples)
    return track

def mixed_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Mixed(seed=seed).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Mixed + Rolling Accuracy", dataset, metric, n_samples)
    return track

def sea_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.SEA(seed=seed).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("SEA + Rolling Accuracy", dataset, metric, n_samples)
    return track

def sine_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.Sine(seed=seed).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("SINE + Rolling Accuracy", dataset, metric, n_samples)
    return track

def stagger_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = synth.STAGGER(seed=seed).take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("STAGGER + Rolling Accuracy", dataset, metric, n_samples)
    return track

def elec2_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = Elec2().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Elec2 + Rolling Accuracy", dataset, metric, n_samples)
    return track

def bananas_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = Bananas().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Bananas + Rolling Accuracy", dataset, metric, n_samples)
    return track

def creditcard_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = CreditCard().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("CreditCard + Rolling Accuracy", dataset, metric, n_samples)
    return track

def higgs_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = Higgs().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Higgs + Rolling Accuracy", dataset, metric, n_samples)
    return track

def imagesegments_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = ImageSegments().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("ImageSegmentation + Rolling Accuracy", dataset, metric, n_samples)
    return track

def insects_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = Insects().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Insects + Rolling Accuracy", dataset, metric, n_samples)
    return track

def maliciousURL_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = MaliciousURL().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("MaliciousURL + Rolling Accuracy", dataset, metric, n_samples)
    return track

def music_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = Music().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Music + Rolling Accuracy", dataset, metric, n_samples)
    return track

def pishing_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = Phishing().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("Phishing + Rolling Accuracy", dataset, metric, n_samples)
    return track

def smsspam_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = SMSSpam().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("SMSSpam + Rolling Accuracy", dataset, metric, n_samples)
    return track

def trec07_rolling_accuracy_track(n_samples=10_000, seed=42):
    dataset = TREC07().take(n_samples)
    metric = Rolling(Accuracy(), window_size=METRIC_ROLLING_WINDOW)
    track = Track("TREC07 + Rolling Accuracy", dataset, metric, n_samples)
    return track
