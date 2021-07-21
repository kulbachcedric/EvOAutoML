from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == '__main__':
    csv_path = Path('../results/evaluation_adaption/AGRAWAL/Agrawal + Accuracy.csv')
    data = pd.read_csv(str(csv_path))
    sns.lineplot(data=data, x=data['step'], y=data['pipe scores'], hue=data['pipe names'], alpha=.5)
    #sns.lineplot(data=data, x=data['step'], y=data['errors'])
    plt.show()