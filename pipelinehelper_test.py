from river import compose, preprocessing, tree
from algorithm import pipelinehelper
from river.evaluate.classification_tracks import random_rbf_track, agrawal_track, anomaly_sine_track, concept_drift_track, hyperplane_track, mixed_track, sea_track, sine_track, stagger_track
from utils import plot_track
import matplotlib.pyplot as plt

tracks = [
    ('Random RBF', random_rbf_track),
    ('AGRAWAL', agrawal_track),
    ('Anomaly Sine', anomaly_sine_track),
    ('Concept Drift', concept_drift_track),
    ('Hyperplane', hyperplane_track),
    ('Mixed', mixed_track),
    ('SEA', sea_track),
    ('Sine', sine_track),
    ('STAGGER', stagger_track)
]

estimator1 = compose.Pipeline(
    preprocessing.StandardScaler(),
    #feature_extraction.PolynomialExtender(),
    tree.HoeffdingTreeClassifier()
)

estimator2 = compose.Pipeline(
    pipelinehelper.PipelineHelperTransformer([
        ('scaler', preprocessing.StandardScaler())
    ]),
    #feature_extraction.PolynomialExtender(),
    ('classifier', tree.HoeffdingTreeClassifier())
)
estimator3 = compose.Pipeline(
    ('scaler', preprocessing.StandardScaler()),
    pipelinehelper.PipelineHelperClassifier([
        ('classifier', tree.HoeffdingTreeClassifier())
    ])
)
if __name__ == '__main__':
    for track_name, track in tracks:
        fig = plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                'Classifier Only' : tree.HoeffdingTreeClassifier(),
                'Without Pipeline': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
                'Pipeline': estimator1,
                'PipelineHelperTransformer' : estimator2,
                'PipelineHelperClassifier': estimator3,
            },
            n_samples=10_000,
            n_checkpoints=100,
            name=track_name,
            verbose=0,
        )
        plt.plot()

