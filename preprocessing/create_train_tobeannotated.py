import pandas as pd
import os

n_samples = 50

all_df = []
for file in os.listdir("../data/"):
    if not file.endswith("_train.csv"): # not shared, contact authors for data access
        continue
    if len(file.split("_")) > 3:
        subreddit = file.split("_")[0]+"_"+file.split("_")[1]
    else:
        subreddit = file.split("_")[0]
    data = pd.read_csv("../data/"+file)

    if len(data) < n_samples:
        n = len(data)
    else:
        n = n_samples

    data = data.sample(n=n, random_state=42)

    all_df.append(data)

final_df = pd.concat(all_df)
# re-index 
final_df.reset_index(drop=True, inplace=True)

# add empty annotation column
final_df["annotation"] = ""

final_df.to_csv("../data/train_tobeannotated.csv", index=True)


