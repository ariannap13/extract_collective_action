import pandas as pd
import os

n_samples = 20

all_df = []
for file in os.listdir("../data/"):
    if not file.endswith("_test.csv"): # not shared, contact authors for data access
        continue
    if len(file.split("_")) > 3:
        subreddit = file.split("_")[0]+"_"+file.split("_")[1]
    else:
        subreddit = file.split("_")[0]
    data = pd.read_csv("../data/"+file)

    comb_data = data.groupby(['link_id','author']).size().reset_index().rename(columns={0:'count'})
    
    if len(comb_data) < n_samples:
        n = len(comb_data)
    else:
        n = n_samples

    comb_data = comb_data.sample(n=n, random_state=42)

    final_df = []
    for i, couple in comb_data.iterrows():
        relevant_data = data[(data['link_id'] == couple['link_id']) & (data['author'] == couple['author'])]
        # sample 1 comment 
        sample = relevant_data.sample(n=1, random_state=42)
        final_df.append(sample)
    final_df = pd.concat(final_df)
    
    all_df.append(final_df)

final_df = pd.concat(all_df)
# re-index 
final_df.reset_index(drop=True, inplace=True)

# add empty annotation column
final_df["annotation"] = ""

final_df.to_csv("../data/test_tobeannotated.csv", index=True)


