from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd

def plot_csv(csv_path:Path, metric_name = 'Accuracy'):
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5), nrows=3, dpi=300 )

    data = pd.read_csv(str(csv_path))

    model_names = list(data['model'].unique())
    model_names.remove('EvoAutoML Leveraging')
    for model_name in model_names:

        step = data.loc[(data['model'] == model_name)]['step']
        error = data.loc[(data['model'] == model_name)]['errors']
        r_time = data.loc[(data['model'] == model_name)]['r_times']
        memory = data.loc[(data['model'] == model_name)]['memories']

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

    result_path = csv_path.parent / f'{csv_path.stem}.pdf'
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(result_path))

if __name__ == '__main__':
    path = Path('../results/classification/evaluation_ensemble/SEA(50).csv')
    plot_csv(path)