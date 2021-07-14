from river import compose, preprocessing, tree, neighbors, naive_bayes, linear_model
from algorithm import pipelinehelper
from algorithm.tpot import OnlineTpotClassifer
from tracks.classification_tracks import random_rbf_track, agrawal_track, anomaly_sine_track, concept_drift_track, hyperplane_track, mixed_track, sea_track, sine_track, stagger_track
from utils import plot_track
import matplotlib.pyplot as plt

tracks = [
    #('Random RBF', random_rbf_track),
    #('AGRAWAL', agrawal_track),
    #('Anomaly Sine', anomaly_sine_track),
    ('Concept Drift', concept_drift_track),
    #('Hyperplane', hyperplane_track),
    #('Mixed', mixed_track),
    #('SEA', sea_track),
    #('Sine', sine_track),
    #('STAGGER', stagger_track)
]

if __name__ == '__main__':
    for track_name, track in tracks:
        fig = plot_track(
            track=track,
            metric_name="Rolling",
            models={
                'TPOT Classifier': OnlineTpotClassifer(1000,classes=[False, True]),
                #'HT Classifier Only' : tree.HoeffdingTreeClassifier(),
                'KNN' : neighbors.KNNADWINClassifier(),
                #'linear' : linear_model.PAClassifier()

            },
            n_samples=10_000,
            n_checkpoints=100,
            name=track_name,
            verbose=0,
        )
        plt.plot()

