import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from datasets.streams import StreamId

SHOW_PLOT = True
SAVE_FIG = True






def get_path(experiment_type, experiment, model, topic):
    return f'../results/{experiment_type}/{experiment}.{model}.{topic}.csv'

def rename_model(model) -> str:
    res = 'Not implemented'
    if model == 'EvolutionaryBestClassifier':
        res = 'EvolutionaryBest'
    elif model == 'BLASTClassifier':
        res = 'BLAST'
    elif model == 'MetaStreamClassifier':
        res = 'MetaStream'
    elif model == 'HoeffdingTreeClassifier':
        res = 'Hoeffding Tree'
    elif model == 'KNNClassifier':
        res = 'k-NN'
    elif model == 'PerceptronMask':
        res = 'Perceptron'
    elif model == 'SGDClassifier':
        res = 'SGD'
    elif model == 'HoeffdingAdaptiveTreeClassifier':
        res = 'Hoeffding Tree Adap.'
    elif model == 'LeveragingBaggingClassifier':
        res = 'Leveraging Bagging'
    elif model == 'OzaBaggingAdwinClassifier':
        res = 'Oza Bagging Adwin'

    return res

def plot_violins_sns(experiment_type,experiment,models,topics):
    plt.close()
    metric = 'mean_acc_[M0]'
    model_metrics = []

    data = list()

    for index, model in enumerate(models):
        topic_metrics = pd.Series()
        for topic in topics:
            path = get_path(experiment_type, experiment, models[index], topic)
            df = pd.read_csv(path, comment='#')
            current_metric_topic = df[[metric]]
            mean_topic = current_metric_topic.mean()
            data.append([rename_model(model), topic, float(mean_topic)])
            topic_metrics = pd.concat([topic_metrics, mean_topic])
        model_metrics.append(topic_metrics.values)

    data = pd.DataFrame(columns=['model', 'topic', 'value'], data=data)

    my_pal = {species: "darkgrey" if species == "EvolutionaryBest" else "gainsboro" for species in data.model.unique()}

    #fig, ax = plt.subplots()
    ax = sns.boxplot(x="value",
                     y="model",
                     data=data,
                     palette=my_pal,
                     showmeans=True,
                     meanprops={"marker":"|",
                       "markerfacecolor":"red",
                       "markeredgecolor":"red",
                      "markersize":"20"}
                     )
    ax.set(ylabel=None)
    ax.set(xlabel='dataset-averaged accuracy')
    plt.gcf().subplots_adjust(left=0.27)

    if SAVE_FIG:
        plt.savefig(f'../results/{experiment_type}/{experiment}_all_models_all_topics_violin.png')
    if SHOW_PLOT:
        plt.show()


def plot_violins(experiment_type, experiment, models, topics):
    plt.close()
    metric = 'mean_acc_[M0]'
    model_metrics = []
    for index, model in enumerate(models):
        topic_metrics = pd.Series()
        for topic in topics:
            path = get_path(experiment_type, experiment, rename_model(models[index]), topic)
            df = pd.read_csv(path, comment='#')
            current_metric_topic = df[[metric]]
            mean_topic = current_metric_topic.mean()
            topic_metrics = pd.concat([topic_metrics, mean_topic])
        model_metrics.append(topic_metrics.values)




    plt.violinplot(model_metrics, showmeans=True, showmedians=True)
    #plt.ylim(0, 1)
    plt.gca().grid(which='both', axis='y', linestyle='dotted')
    plt.xticks(np.arange(1, len(models) + 1), models, rotation=45)
    # plt.title('AutoML Models Accuracy (Dataset-averaged)', pad=26)
    plt.title(f'{experiment_type.title()} Models Accuracy (Dataset-averaged)', pad=26)
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(f'../results/{experiment_type}/{experiment}_all_models_all_topics_violin.png')
    if SHOW_PLOT:
        plt.show()




if __name__ == "__main__":
    # experiments = ['batch']
    experiments = ['online']
    # experiment_type = 'batch_learning'
    experiment_types = [
        'experiment_1',
        'experiment_25',
        'experiment_50',
        'experiment_100',
        'experiment_250',
        'experiment_500',
        'experiment_1000',
        'experiment_1000'
    ]
    topics = [
        #StreamId.agrawal_gen.name,
        #StreamId.stagger_gen.name,
        StreamId.hyperplane_gen.name,
        StreamId.led_gen.name,
        StreamId.rbf_gen.name,
        #StreamId.sea_gen.name,
        StreamId.covtype.name,
        StreamId.elec.name,
        StreamId.pokerhand.name,
        StreamId.conceptdrift_sea.name,
        StreamId.conceptdrift_agrawal.name,
        StreamId.conceptdrift_stagger.name
    ]
    batch_models = [
        'RandomForestClassifier',
        'DecisionTreeClassifier',
        'KNeighborsClassifier',
        'LinearSVC',
        'TPOTClassifier'
    ]
    online_models = [
        'EvolutionaryBestClassifier',
        'BLASTClassifier',
        'MetaStreamClassifier',
        'HoeffdingTreeClassifier',
        'KNNClassifier',
        'PerceptronMask',
        'SGDClassifier',
        'HoeffdingAdaptiveTreeClassifier',
        'LeveragingBaggingClassifier',
        'OzaBaggingAdwinClassifier'
    ]

    plot_grouped = False
    for experiment_type in experiment_types:
        for experiment in experiments:
            if experiment == 'batch':
                models = batch_models
            else:
                models = online_models

            demos = models
            demo_types = experiment

            print('Plotting violins', experiment_type, experiment, models, topics)
            plot_violins_sns(experiment_type=experiment_type, experiment=experiment, models=models, topics=topics)
