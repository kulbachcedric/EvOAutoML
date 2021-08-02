from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    evaluation_dir = Path("../results/evaluation_single")

    data = pd.DataFrame()
    for idx, csv_path in enumerate(evaluation_dir.glob('./*.csv')):
        if idx == 0:
            data = pd.read_csv(str(csv_path))
        else:
            data.append(str(csv_path))

    data.to_csv(str(evaluation_dir)+"/merged.csv")
