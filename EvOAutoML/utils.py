from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from river.evaluate import Track
from tqdm import tqdm
import mlflow



def plot_track(track : Track,
               metric_name,
               models,
               n_samples,
               n_checkpoints,
               result_path:Path = None,
               verbose=1):

    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5), nrows=3, dpi=300 )
    print(f'Track name: {track(n_samples=1,seed=42).name}')
    result_data = {
        'step': [],
        'model' : [],
        'errors' : [],
        'r_times' : [],
        'memories' : []
    }

    for model_name, model in models.items():
        if verbose > 1:
            print(f'Evaluating {model_name}')
        step = []
        error = []
        r_time = []
        memory = []
        if verbose < 1:
            disable = True
        else:
            disable = False
        for checkpoint in tqdm(track(n_samples=n_samples, seed=42).run(model, n_checkpoints),disable=disable):
            step.append(checkpoint["Step"])
            error.append(checkpoint[metric_name])
            # Convert timedelta object into seconds
            r_time.append(checkpoint["Time"].total_seconds())
            # Make sure the memory measurements are in MB
            raw_memory, unit = float(checkpoint["Memory"][:-3]), checkpoint["Memory"][-2:]
            memory.append(raw_memory * 2**-10 if unit == 'KB' else raw_memory)
        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        ax[0].plot(step, error, label=model_name,linewidth=.6)
        ax[0].set_ylabel(metric_name)
        #ax[0].set_ylabel('Rolling 100\n Accuracy')

        ax[1].plot(step, r_time, label=model_name,linewidth=.6)
        ax[1].set_ylabel('Time (seconds)')

        ax[2].plot(step, memory, label=model_name,linewidth=.6)
        ax[2].set_ylabel('Memory (MB)')
        ax[2].set_xlabel('Instances')

        result_data['step'].extend(step)
        result_data['model'].extend(len(step)*[model_name])
        result_data['errors'].extend(error)
        result_data['r_times'].extend(r_time)
        result_data['memories'].extend(memory)

    plt.legend()
    plt.tight_layout()
    df = pd.DataFrame(result_data)
    if result_path is not None:
        result_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(result_path / f'{track().name}.pdf'))
        df.to_csv(str(result_path / f'{track().name}.csv'))

    return df

def evaluate_track(
    track: Track,
    metric_name,
    model_tuple,
    n_samples,
    n_checkpoints,
    verbose = 1):
    track_name = track(n_samples=1, seed=42).name
    result_data = {
        'step': [],
        'model': [],
        'dataset' : [],
        'errors': [],
        'r_times': [],
        'memories': []
    }

    model_name, model = model_tuple

    if verbose > 1:
        print(f'Evaluating {model_name}')
    step = []
    error = []
    r_time = []
    memory = []
    if verbose < 1:
        disable = True
    else:
        disable = False
    for checkpoint in tqdm(track(n_samples=n_samples, seed=42).run(model, n_checkpoints),disable=disable):
        step.append(checkpoint["Step"])
        error.append(checkpoint[metric_name])
        # Convert timedelta object into seconds
        r_time.append(checkpoint["Time"].total_seconds())
        # Make sure the memory measurements are in MB
        raw_memory, unit = float(checkpoint["Memory"][:-3]), checkpoint["Memory"][-2:]
        memory.append(raw_memory * 2**-10 if unit == 'KB' else raw_memory)

    result_data['step'].extend(step)
    result_data['model'].extend(len(step) * [model_name])
    result_data['dataset'].extend(len(step) * [track_name])
    result_data['errors'].extend(error)
    result_data['r_times'].extend(r_time)
    result_data['memories'].extend(memory)
    df = pd.DataFrame(result_data)

    return df

def evaluate_track_mlflow(
    track: Track,
    metric_name,
    model_tuple,
    n_samples,
    n_checkpoints,
    verbose = 1):

    remote_server_uri = "http://ipe-mufflon.fzi.de:5000"  # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    #mlflow.delete_experiment("Online AutoML")
    mlflow.set_experiment("EvOAutoML 1")
    track_name = track(n_samples=1, seed=42).name
    result_data = {
        'step': [],
        'model': [],
        'dataset': [],
        'errors': [],
        'r_times': [],
        'memories': []
    }

    model_name, model = model_tuple

    if verbose > 1:
        print(f'Evaluating {model_name}')
    step = []
    error = []
    r_time = []
    memory = []
    if verbose < 1:
        disable = True
    else:
        disable = False



    with mlflow.start_run():
        mlflow.log_param('model',model_name)
        mlflow.log_param('dataset', track_name)
        for checkpoint in tqdm(track(n_samples=n_samples, seed=42).run(model, n_checkpoints),disable=disable):
            step.append(checkpoint["Step"])
           #mlflow.log_metric('step',checkpoint["Step"], step=checkpoint["Step"])

            error.append(checkpoint[metric_name])
            mlflow.log_metric(metric_name,checkpoint[metric_name], step=checkpoint["Step"])

            # Convert timedelta object into seconds
            r_time.append(checkpoint["Time"].total_seconds())
            mlflow.log_metric('r_times',checkpoint["Time"].total_seconds(),step=checkpoint["Step"])

            # Make sure the memory measurements are in MB
            raw_memory, unit = float(checkpoint["Memory"][:-3]), checkpoint["Memory"][-2:]
            memory.append(raw_memory * 2**-10 if unit == 'KB' else raw_memory)
            mlflow.log_metric("memories", raw_memory * 2**-10 if unit == 'KB' else raw_memory, step=checkpoint["Step"])

        result_data['step'].extend(step)
        result_data['model'].extend(len(step)*[model_name])
        result_data['dataset'].extend(len(step)*[track_name])
        result_data['errors'].extend(error)
        result_data['r_times'].extend(r_time)
        result_data['memories'].extend(memory)

        df = pd.DataFrame(result_data)

        return df
