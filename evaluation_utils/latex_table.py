from pathlib import Path
import pandas as pd

def get_dataset_row(csv_path:Path):
    data = pd.read_csv(str(csv_path))

    models = data['model'].unique()
    max_step = data['step'].max()

    new_data = {}

    ranks = data[(data['step'] == max_step)][['model','errors']]
    ranks['rank'] = ranks['errors'].rank(method='max', ascending=False)


    new_data['Dataset'] = [csv_path.stem]
    for model in models:
        # Get Time
        time = "%.3f" % float(data[(data['model']==model) & (data['step'] == max_step)]['r_times'])
        accuracy = "%.3f" % float(data[(data['model']==model) & (data['step'] == max_step)]['errors'])
        std = "%.3f" % float(data[(data['model']==model)]['errors'].std())
        memory = "%.3f" % float(data[(data['model']==model)]['memories'].mean())

        new_data[f'{model}__Time'] = [time]
        new_data[f'{model}__Accuracy'] = [accuracy]
        new_data[f'{model}__std'] = [std]
        new_data[f'{model}__Memory'] = [memory]
        new_data[f'{model}__Rank'] = [float(ranks[(data['model']==model)]['rank'])]
    return_data = pd.DataFrame(new_data)
    return return_data


if __name__ == '__main__':

    evaluation = 'evaluation_ensemble'
    #evaluation = 'evaluation_single'
    evaluation_dir = Path(f'../results/classification/{evaluation}')

    data = pd.DataFrame()
    for idx, csv_path in enumerate(evaluation_dir.glob('./*.csv')):
        dataset_row = get_dataset_row(csv_path)
        data = data.append(dataset_row)

    data.columns = data.columns.str.split('__', expand=True)
    data.to_excel(f'{evaluation}.xlsx')