import itertools
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm


def visualize_adaption(evaluation_dir:Path):
    for experiment_dir in tqdm(evaluation_dir.glob('./*')):
        for csv_dir in experiment_dir.glob('./*.csv'):
            data = pd.read_csv(str(csv_dir))
            f, ax = plt.subplots(1, 1)
            palette = itertools.cycle(sns.color_palette())
            ax.plot(data['step'], data['errors'], color="red", label="EvO AutoML", linestyle="-")

            for i in range(10):
                filtered_data = data[(data['pipe names'] == f"Individual {i}")]
                ax.plot(filtered_data['step'], filtered_data['pipe scores'],
                        color=next(palette),
                        label=f"Individual {i}",
                        linestyle="-",
                        alpha=.8,
                        linewidth=.6
                        )
            ax.legend()
            ax.set_title(csv_dir.parent.name)
            ax.grid(True)
            ax.set_xlabel('Instances')
            ax.set_ylabel('Accuracy')

            #filtered_data = data[['step', 'errors']]


            plt.savefig(str(experiment_dir)+'/population_performances.pdf')
            plt.show()
            plt.close()


if __name__ == '__main__':

    evaluation_dir = Path(f'../results/evaluation_adaption')
    visualize_adaption(evaluation_dir)