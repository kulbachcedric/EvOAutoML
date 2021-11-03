from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
#import matplotlib as mpl

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
#mpl.rc('font', family='serif')

def plot_csv(csv_path:Path, metric_name = 'Accuracy'):
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 4), nrows=2, dpi=300 )

    data = pd.read_csv(str(csv_path))



    data = data.replace('EvoAutoML Bagging', value="EvoAutoML")
    data = data.replace('Leveraging Bagging', value="LB")
    data = data.replace('Bagging', value="OB")
    data = data.replace('Hoeffding Tree', value="HT")
    data = data.replace('GaussianNB', value="GNB")


    model_names = list(data['model'].unique())
    #model_names.remove('EvoAutoML Leveraging')
    #model_names.remove('Ada Boost')
    for model_name in model_names:

        step = data.loc[(data['model'] == model_name)]['step']
        error = data.loc[(data['model'] == model_name)]['errors']
        r_time = data.loc[(data['model'] == model_name)]['r_times']
        memory = data.loc[(data['model'] == model_name)]['memories']

        ax[0].grid(True)
        ax[1].grid(True)
        #ax[2].grid(True)

        ax[0].plot(step, error, label=model_name,linewidth=.6)
        ax[0].set_ylabel(metric_name)
        #ax[0].set_ylim([.9,1.0])
        #ax[0].set_ylabel('Rolling 100\n Accuracy')

        ax[1].plot(step, r_time, label=model_name,linewidth=.6)
        ax[1].set_ylabel('Time (seconds)')

        #ax[2].plot(step, memory, label=model_name,linewidth=.6)
        #ax[2].set_ylabel('Memory (MB)')
        ax[1].set_xlabel('Instances')
    result_path = csv_path.parent / f'{csv_path.stem}.pdf'
    #plt.legend()
    move_legend_below_graph(ax,4)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    #plt.show()
    plt.savefig(str(result_path))

def move_legend_below_graph(axes, ncol: int):
    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=ncol)
    plt.tight_layout()

if __name__ == '__main__':
    path = Path('../results/classification/evaluation_ensemble/Covtype_all.csv')
    plot_csv(path)