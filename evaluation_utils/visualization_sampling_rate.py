import itertools
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    sampling_rates = [
        #1,
        #5,
        10,
        25,
        50,
        100,
        #150,
        #200,
        250,
        500,
        #750,
        1000
    ]

    dir = Path(f'../results/evaluation_adaption')

    evaluation_file = Path(str(dir) + '.xlsx')
    data = pd.read_excel(str(evaluation_file))
    for dataset in data['track'].unique():
        f, ax = plt.subplots(1, 1)
        palette = itertools.cycle(sns.color_palette("rocket"))
        for idx, i in enumerate(sampling_rates):
            data_filtered = data[(data['sampling_rate'] == i) &
                                 (data['track'] == dataset) &
                                 (data['model'] == 'EvoAutoML')
                                 ]
            ax.plot(data_filtered['step'], data_filtered['errors'],
                    color=next(palette),
                    label=f"{i}",
                    linestyle="-",
                    # alpha=1-(idx/len(sampling_rates)),
                    linewidth=.6
                    )
        ax.legend(title='Sampling Rate')
        ax.grid(True)
        ax.set_xlabel('Instances')
        ax.set_ylabel('Accuracy')
        ax.set_title(dataset)
        result_path = dir
        plt.savefig(str(result_path) + f'/{dataset}.pdf')
        # plt.show()
        plt.close()