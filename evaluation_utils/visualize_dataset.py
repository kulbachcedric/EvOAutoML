from pathlib import Path
import pandas as pd
from evaluation_utils.visualize_dataset_csv import plot_dataframe

if __name__ == '__main__':

    evaluation_dir = Path('../results/classification/evaluation_ensemble/ensemble_evaluation')
    for idx, dataset_path in enumerate(evaluation_dir.glob('./*')):
        if dataset_path.is_dir():
            print(f'Plotting: {dataset_path.stem}')
            result_path = evaluation_dir / f'{dataset_path.stem}.pdf'
            df = pd.DataFrame()
            for idx, csv_path in enumerate(dataset_path.glob('./*.csv')):
                if idx == 0:
                    df = pd.read_csv(str(csv_path))
                    # data['dataset'] = csv_path.stem
                else:
                    data_new = pd.read_csv(str(csv_path))
                    # data_new['dataset'] = csv_path.stem
                    df = df.append(data_new)

            plot_dataframe(df = df, result_path=result_path)