from river import neighbors
from EvOAutoML.tpot import OnlineTpotClassifer
from EvOAutoML.tracks.classification_tracks import concept_drift_accuracy_track
from EvOAutoML.utils import plot_track
import matplotlib.pyplot as plt

tracks = [
    #('Random RBF', random_rbf_track),
    #('AGRAWAL', agrawal_track),
    #('Anomaly Sine', anomaly_sine_track),
    ('Concept Drift', concept_drift_accuracy_track),
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

