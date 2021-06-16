from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.style as style

from datasets.streams import StreamId

style.use('seaborn-poster')

SHOW_PLOT = False
SAVE_FIG = True


def plot_topics(experiment_type, experiment, model, topics):
    mean_acc = pd.DataFrame()
    mean_kappa = pd.DataFrame()
    current_acc = pd.DataFrame()
    current_kappa = pd.DataFrame()

    for topic in topics:
        print(topic)
        path = get_path(experiment_type, experiment, model, topic)
        df = pd.read_csv(str(path), comment='#')
        x = df[['id']]
        mean_acc = pd.concat([mean_acc, df[['mean_acc_[M0]']]], axis=1)
        mean_kappa = pd.concat([mean_kappa, df[['mean_kappa_[M0]']]], axis=1)
        current_acc = pd.concat([current_acc, df[['current_acc_[M0]']]], axis=1)
        current_kappa = pd.concat([current_kappa, df[['current_kappa_[M0]']]], axis=1)

    plt.close()
    plt.title(f'{model} Model Predictive Accuracy', pad=26)
    plt.ylabel('Mean predictive accuracy')
    plt.xlabel('Number of samples')
    plt.plot(x, mean_acc)
    # ax.plot(x, mean_kappa)
    # ax.plot(x, current_acc)
    # ax.plot(x, current_kappa)
    plt.ylim(ymin=0.45, ymax=1.05)
    plt.legend(topics)
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(f'../results/{experiment_type}/{experiment}_{model}_{len(topics)}_topics.png')
    if SHOW_PLOT:
        plt.show()


def plot_topics_grouped(demo, demo_type, models, topics):
    plt.close()
    f, subplots = plt.subplots(len(models))
    f.set_size_inches(10, 20)
    f.subplots_adjust(top=0.94, hspace=0.4)
    f.suptitle('Models Predictive Accuracy', fontsize=26)

    for index, subplot in enumerate(subplots):
        mean_acc = pd.DataFrame()
        mean_kappa = pd.DataFrame()
        current_acc = pd.DataFrame()
        current_kappa = pd.DataFrame()
        for topic in topics:
            print(topic)
            path = get_path(demo, demo_type, models[index], topic)
            df = pd.read_csv(path, comment='#')
            x = df[['id']]
            mean_acc = pd.concat([mean_acc, df[['mean_acc_[M0]']]], axis=1)
            mean_kappa = pd.concat([mean_kappa, df[['mean_kappa_[M0]']]], axis=1)
            current_acc = pd.concat([current_acc, df[['current_acc_[M0]']]], axis=1)
            current_kappa = pd.concat([current_kappa, df[['current_kappa_[M0]']]], axis=1)

        subplot.set_title(models[index])
        subplot.set_ylabel('Mean accuracy')
        if index == len(subplots) - 1:
            subplot.set_xlabel('Number of samples')
        subplot.plot(x, mean_acc)
        # subplot.plot(x, mean_kappa)
        # subplot.plot(x, current_acc)
        # subplot.plot(x, current_kappa)
        # subplot.set_ylim(ymax=1.05)

    plt.legend(topics, loc='center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=3)

    if SAVE_FIG:
        plt.savefig(f'../{demo}/results/{demo_type}_all_models_all_topics.png')
    if SHOW_PLOT:
        plt.show()

def get_path(experiment_type, experiment, model, topic):
    return Path(f'../results/{experiment_type}/{experiment}.{model}.{topic}.csv')


if __name__ == "__main__":
    #experiments = ['batch']
    experiments = ['online']
    #experiment_type = 'batch_learning'
    experiment_types = [
        #'experiment_1',
        'experiment_25',
        'experiment_50',
        'experiment_100',
        'experiment_250',
        'experiment_500',
        'experiment_1000',
        'experiment_2000',
    ]
    topics = [

        #StreamId.agrawal_gen.name,
        ##StreamId.stagger_gen.name,
        StreamId.hyperplane_gen.name,
        StreamId.led_gen.name,
        StreamId.rbf_gen.name,
        ##StreamId.sea_gen.name,
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

            if plot_grouped:
                print('Plotting grouped', experiment_type, experiment, models, topics)
                plot_topics_grouped(experiment_type, experiment, models, topics)
            else:
                for model in models:
                    print('Plotting topics', experiment, experiment_type, model, topics)
                    plot_topics(experiment_type, experiment, model, topics)
