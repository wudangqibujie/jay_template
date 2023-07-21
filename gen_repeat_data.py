import pandas as pd
from pathlib import Path


data = pd.read_csv("mini_data/tabular_csv/movielens_sample.txt")

train_data_folder = Path("../data/movielens/train")
valid_data_folder = Path("../data/movielens/valid")
test_data_folder = Path("../data/movielens/test")

for ix in range(100):
    data.to_csv(train_data_folder / f"{ix}.csv", index=False)

for ix in range(100, 150):
    data.to_csv(valid_data_folder / f"{ix}.csv", index=False)

for ix in range(150, 200):
    data.to_csv(test_data_folder / f"{ix}.csv", index=False)


