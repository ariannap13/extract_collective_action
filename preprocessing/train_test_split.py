import os
from sklearn.model_selection import train_test_split
import pandas as pd

data_dir = "../data/"

for sub_file in os.listdir(data_dir):
    if sub_file.endswith("_whole.csv"): # data files gathered from the full Reddit collection, not shared.
        if len(sub_file.split("_")) > 4:
            sub_name = "_".join([sub_file.split("_")[0], sub_file.split("_")[1]])
        else:
            sub_name = sub_file.split("_")[0]
        df = pd.read_csv(os.path.join(data_dir, sub_file))
        train, test = train_test_split(df, test_size=0.05, random_state=42)
        train.to_csv(os.path.join(data_dir+"test_train/train/", sub_name + "_train.csv"))
        test.to_csv(os.path.join(data_dir+"test_train/test/", sub_name + "_test.csv"))
