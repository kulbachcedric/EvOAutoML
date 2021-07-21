from pathlib import Path

from river.evaluate import Track
from tqdm import tqdm
import pandas as pd
from EvOAutoML.config import CLASSIFICATION_TRACKS, AUTOML_PIPELINE, PARAM_GRID
from matplotlib import pyplot as plt

from EvOAutoML.oaml import EvolutionaryBestClassifier
from EvOAutoML.tracks.evo_classification_tracks import evo_random_rbf_track, evo_agrawal_track, evo_anomaly_sine_track, \
    evo_concept_drift_track, evo_hyperplane_track, evo_mixed_track, evo_sea_track, evo_sine_track, evo_stagger_track, \
    EvoTrack


def plot_track(track : EvoTrack,
               metric_name,
               models,
               n_samples,
               n_checkpoints,
               result_path:Path = None,
               verbose=1):
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5), nrows=3, dpi=300 )

    result_data = {
        'step' : [],
        'model' : [],
        'errors' : [],
        'r_times' : [],
        'memories' : [],
        'pipe names' : [],
        'pipe scores' : []
    }

    for model_name, model in models.items():
        if verbose > 1:
            print(f'Evaluating {model_name}')
        step = []
        error = []
        r_time = []
        memory = []
        pipe_name = []
        pipe_performance = []
        if verbose < 1:
            disable = True
        else:
            disable = False
        for checkpoint in tqdm(track(n_samples=n_samples, seed=42).run(model, n_checkpoints),disable=disable):
            step.extend(checkpoint["Step"])
            error.extend(checkpoint[metric_name])
            # Convert timedelta object into seconds
            r_time.extend([t.total_seconds() for t in checkpoint["Time"]])
            # Make sure the memory measurements are in MB
            for mem in checkpoint["Memory"]:
                raw_memory, unit = float(mem[:-3]), mem[-2:]
                memory.append(raw_memory * 2**-10 if unit == 'KB' else raw_memory)
            pipe_performance.extend(checkpoint['Model Performance'])
            pipe_name.extend(checkpoint['Name'])

        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        ax[0].plot(step, error, label=model_name)
        ax[0].set_ylabel(metric_name)
        #ax[0].set_ylabel('Rolling 100\n Accuracy')

        ax[1].plot(step, r_time, label=model_name)
        ax[1].set_ylabel('Time (seconds)')

        ax[2].plot(step, memory, label=model_name)
        ax[2].set_ylabel('Memory (MB)')
        ax[2].set_xlabel('Instances')

        result_data['step'].extend(step)
        result_data['model'].extend(len(step)*[model_name])
        result_data['errors'].extend(error)
        result_data['r_times'].extend(r_time)
        result_data['memories'].extend(memory)
        result_data['pipe names'].extend(pipe_name)
        result_data['pipe scores'].extend(pipe_performance)

    plt.legend()
    plt.tight_layout()
    #plt.show()
    df = pd.DataFrame(result_data)
    if result_path is not None:
        result_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(result_path / f'{track().name}.pdf'))
        df.to_csv(str(result_path / f'{track().name}.csv'))

    return df




if __name__ == '__main__':
    EVO_CLASSIFICATION_TRACKS = [
        ('Random RBF', evo_random_rbf_track),
        ('AGRAWAL', evo_agrawal_track),
        ('Anomaly Sine', evo_anomaly_sine_track),
        ('Concept Drift', evo_concept_drift_track),
        ('Hyperplane', evo_hyperplane_track),
        ('Mixed', evo_mixed_track),
        ('SEA', evo_sea_track),
        ('Sine', evo_sine_track),
        ('STAGGER', evo_stagger_track)
    ]


    for track_name, track in EVO_CLASSIFICATION_TRACKS:
        data = plot_track(
            track=track,
            metric_name='Accuracy',
            models={
                'EvoAutoML': EvolutionaryBestClassifier(population_size=5, estimator=AUTOML_PIPELINE, param_grid=PARAM_GRID, sampling_rate=100),
            },
            n_samples=10_000,
            n_checkpoints=1000,
            result_path=Path(f'./results/evaluation_adaption/{track_name}'),
            verbose=2
                   )