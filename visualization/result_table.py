from pathlib import Path

import pandas as pd

from datasets.streams import StreamId


def get_path(experiment_type, experiment, model, topic):
    return Path(f'../results/{experiment_type}/{experiment}.{model}.{topic}.csv')

def create_result_table(experiment_type,experiment,models,topics, data = []):

    data1 = []
    columns = ['dataset', 'algorithm', 'experiment_type', 'metric', 'score']
    for topic in topics:
        print(topic)
        for model in models:
            path = get_path(experiment_type, experiment, model, topic)
            df = pd.read_csv(str(path), comment='#')
            dataset = topic
            algorithm = model
            last_mean_accuracy = df[['mean_acc_[M0]']].iloc[[-1]].values[0][0]
            last_mean_kappa = df[['mean_kappa_[M0]']].iloc[[-1]].values[0][0]
            last_mean_kappa_t = df[['mean_kappa_t_[M0]']].iloc[[-1]].values[0][0]
            last_mean_kappa_m = df[['mean_kappa_m_[M0]']].iloc[[-1]].values[0][0]
            last_mean_precision = df[['mean_precision_[M0]']].iloc[[-1]].values[0][0]
            last_mean_recall = df[['mean_recall_[M0]']].iloc[[-1]].values[0][0]
            last_mean_f1 = df[['mean_f1_[M0]']].iloc[[-1]].values[0][0]
            max_model_size = df[['model_size_[M0]']].max()[0]
            mean_training_time = df[['training_time_[M0]']].mean()[0]
            mean_testing_time = df[['testing_time_[M0]']].mean()[0]
            total_running_time = df[['total_running_time_[M0]']].iloc[[-1]].values[0][0]

            data.append([dataset,algorithm,experiment_type,'accuracy',last_mean_accuracy])
            data.append([dataset, algorithm,experiment_type, 'kappa', last_mean_kappa])
            data.append([dataset, algorithm,experiment_type, 'kappa_t', last_mean_kappa_t])
            data.append([dataset, algorithm,experiment_type, 'kappa_m', last_mean_kappa_m])
            data.append([dataset, algorithm,experiment_type, 'precision', last_mean_precision])
            data.append([dataset, algorithm,experiment_type, 'recall', last_mean_recall])
            data.append([dataset, algorithm,experiment_type, 'f1', last_mean_f1])
            data.append([dataset, algorithm,experiment_type, 'model_size', max_model_size])
            data.append([dataset, algorithm,experiment_type, 'training_time', mean_training_time])
            data.append([dataset, algorithm,experiment_type, 'testing_time', mean_testing_time])
            data.append([dataset, algorithm,experiment_type, 'total_running_time', total_running_time])

            data1.append([dataset,algorithm,experiment_type,'accuracy',last_mean_accuracy])
            data1.append([dataset, algorithm,experiment_type, 'kappa', last_mean_kappa])
            data1.append([dataset, algorithm,experiment_type, 'kappa_t', last_mean_kappa_t])
            data1.append([dataset, algorithm,experiment_type, 'kappa_m', last_mean_kappa_m])
            data1.append([dataset, algorithm,experiment_type, 'precision', last_mean_precision])
            data1.append([dataset, algorithm,experiment_type, 'recall', last_mean_recall])
            data1.append([dataset, algorithm,experiment_type, 'f1', last_mean_f1])
            data1.append([dataset, algorithm,experiment_type, 'model_size', max_model_size])
            data1.append([dataset, algorithm,experiment_type, 'training_time', mean_training_time])
            data1.append([dataset, algorithm,experiment_type, 'testing_time', mean_testing_time])
            data1.append([dataset, algorithm,experiment_type, 'total_running_time', total_running_time])

    df = pd.DataFrame(data=data, columns=columns)
    df1 = pd.DataFrame(data=data1, columns=columns)
    df.to_excel(f'test.xlsx')
    df1.to_excel(f'test.{experiment_type}.xlsx')

    return data

if __name__ == '__main__':
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
    data = []
    for experiment_type in experiment_types:
        for experiment in experiments:
            if experiment == 'batch':
                models = batch_models
            else:
                models = online_models

            create_result_table(experiment_type,experiment,models,topics, data=data)
