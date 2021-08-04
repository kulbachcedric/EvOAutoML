import itertools
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from river import ensemble
from tqdm import tqdm

from EvOAutoML.classification import EvolutionaryBestClassifier
from EvOAutoML.config import AUTOML_CLASSIFICATION_PIPELINE, CLASSIFICATION_PARAM_GRID, POPULATION_SIZE, N_SAMPLES, \
    N_CHECKPOINTS
import pandas as pd

from EvOAutoML.tracks.evo_classification_tracks import EvoTrack, evo_random_rbf_track, evo_agrawal_track, \
    evo_anomaly_sine_track, evo_concept_drift_track, evo_hyperplane_track, evo_mixed_track, evo_sea_track, \
    evo_sine_track, evo_stagger_track
from matplotlib import pyplot as plt

from evaluation_utils.visualization_adaption import visualize_adaption

folder_name = 'evaluation_adaption'


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

def evaluate_sampling_rate(sampling_rate:int,track_tuple:Tuple):
    track_name = track_tuple[0]
    track = track_tuple[1]
    data = plot_track(
        track=track,
        metric_name="Accuracy",
        models={
            'EvoAutoML': EvolutionaryBestClassifier(population_size=POPULATION_SIZE, estimator=AUTOML_CLASSIFICATION_PIPELINE, param_grid=CLASSIFICATION_PARAM_GRID, sampling_rate=sampling_rate),
        },
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        result_path=Path(f'./results/classification/{folder_name}/{track_name}_{sampling_rate}'),
        verbose=2
    )
    data['sampling_rate'] = len(data)*[sampling_rate]
    data['track'] = len(data)*[track_name]
    return data


if __name__ == '__main__':
    sampling_rates = [10,50,100,250,500,750,1000]
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

    testing_configurations = list(itertools.product(sampling_rates,EVO_CLASSIFICATION_TRACKS))

    pool = Pool(60)  # Create a multiprocessing Pool
    output = pool.starmap(evaluate_sampling_rate, testing_configurations)
    result_data = pd.concat(output)

    #t = evaluate_sampling_rate(250,EVO_CLASSIFICATION_TRACKS[2])
    #result_data = t

    result_path = Path(f'./results')
    result_path.mkdir(parents=True, exist_ok=True)
    result_path = result_path / f'{folder_name}.xlsx'
    result_data.to_excel(str(result_path))
    visualize_adaption(Path(f'./results/classification/{folder_name}'))