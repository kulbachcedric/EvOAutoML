from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from river.evaluate import Track
import pandas as pd


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
