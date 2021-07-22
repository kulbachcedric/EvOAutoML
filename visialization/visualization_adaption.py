import itertools
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from EvOAutoML.config import CLASSIFICATION_TRACKS

if __name__ == '__main__':

    evaluation_dir = Path(f'../results/evaluation_adaption_sampling_rate')
    for experiment_dir in tqdm(evaluation_dir.glob('./*')):
        for csv_dir in experiment_dir.glob('./*.csv'):
            data = pd.read_csv(str(csv_dir))
            f, ax = plt.subplots(1, 1)
            palette = itertools.cycle(sns.color_palette())
            filtered_data = filtered_data.drop_duplicates()
            ax.plot(data['step'], data['errors'], color="red", label="EvO AutoML", linestyle="-")

            for i in range(5):
                filtered_data = data[(data['pipe names'] == f"Individual {i}")]
                ax.plot(filtered_data['step'], filtered_data['pipe scores'], color=next(palette), label=f"Individual {i}", linestyle="-", alpha=.6)
            ax.legend()

            #filtered_data = data[['step', 'errors']]


            plt.savefig(str(experiment_dir)+'/population_performances.pdf')
            plt.close()