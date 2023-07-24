import pandas as pd
from pathlib import Path

csv_files = [
    "fr_item_train",
    "fr_item_test",
    "fr_user_train",
    "fr_user_test"
]

data_folders = [
    # Path(r"D:\迅雷下载\ali_dataset\NL"),
    # Path(r"D:\迅雷下载\ali_dataset\ES"),
    Path(r"D:\迅雷下载\ali_dataset\FR"),
    # Path(r"D:\迅雷下载\ali_dataset\us")
]

for data_folder in data_folders:
    for csv_file in csv_files:
        columns = [str(i) for i in range(2, 50)] if 'item' in csv_file else [str(i) for i in range(2, 34)]
        df_iter = pd.read_csv(data_folder / f'{csv_file}.csv', chunksize=5000, header=None, names=['pv-id'] + columns)
        d = dict()
        flag = 0
        df_write = pd.DataFrame()
        for df in df_iter:
            df = df[df['pv-id'] % 10 == 1]
            df_write = pd.concat([df_write, df])
            flag += 1
            if flag % 100 == 0:
                print(flag * 3200, df_write.shape, data_folder, csv_file)
        df_write.to_csv(data_folder / f'{csv_file}_bucket_1.csv', index=False)