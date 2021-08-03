from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    evaluation_dir = Path("../results/evaluation_ensemble")

    data = pd.DataFrame()
    for idx, csv_path in enumerate(evaluation_dir.glob('./*.csv')):
        if idx == 0:
            data = pd.read_csv(str(csv_path))
            data['dataset'] = csv_path.stem
        else:
            data_new = pd.read_csv(str(csv_path))
            data_new['dataset'] = csv_path.stem
            data = data.append(data_new)

    data.to_excel(str(evaluation_dir)+"/merged.xlsx")
