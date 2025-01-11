import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score)
from tqdm import tqdm

labels = ["Problem-Solution", "Call-to-action", "Intention", "Execution", "None"]

labels2id = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3, "None": 4}
id2labels = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution", 4: "None"}

dataset = "test" # "test" or "topic" or "GWSD" or "no_action" or "sub_action" or "no_random" or "sub_random" or "downstream" or "debates"

if dataset!="test":
    dataset_tag = f"_{dataset}"
else:
    dataset_tag = ""

if dataset == "downstream": 
    dataset_type = "climate_change" 
    dataset_tag += f"_{dataset_type}"

# Open first layer predictions
predicted_data = pd.read_csv(f"../data/predictions/predictions_roberta_simplified_synthetic_weights{dataset_tag}.csv")

if dataset == "test":
    data = pd.read_csv("../data/test_set.csv")
    data["text"] = data["ActionFocusedText"]

elif dataset == "downstream":
    if dataset_type == "climate_change":
        data = pd.read_csv(f"../data/comments_top2k_2018_climate_change.csv") # not shared, contact the authors for access

        data["CommentID"] = data["CommentID"]

data = data[data["CommentID"].isin(predicted_data["CommentID"])]

# sort both dataframes by id
data = data.sort_values(by="CommentID")
predicted_data = predicted_data.sort_values(by="CommentID")

# merge both dataframes
data_merge = pd.merge(data, predicted_data, on="CommentID")

# Open second layer predictions
data_second = pd.read_csv(f"../data/predictions/predictions_roberta_synthetic_weights_layered{dataset_tag}.csv")

# Create a dictionary for fast lookup from data_second
id_to_prediction = dict(zip(data_second["CommentID"], data_second["predictions"]))

# Enable tqdm for pandas apply
tqdm.pandas()

# Vectorized operation to compute final_pred
data_merge["final_pred"] = data_merge.progress_apply(
    lambda row: 4 if row["predictions"] == 1 else id_to_prediction.get(row["CommentID"], None),
    axis=1
)

# drop _y columns from the merge
data_merge = data_merge.drop(columns=[col for col in data_merge.columns if col.endswith("_y")])
# rename _x columns to remove the _x
data_merge = data_merge.rename(columns={col: col.replace("_x", "") for col in data_merge.columns})

# save the final predictions
data_merge.to_csv(f"../data/predictions/predictions_roberta_synthetic_weights_layered_final{dataset_tag}.csv", index=False)

y_true_overall = [labels2id[label] for label in data_merge["Label"].values]
y_pred_overall = data_merge["final_pred"].values

# Compute ROC-AUC, F1, Precision, Recall overall
list_scores = []
for label in id2labels:
    y_true = [1 if y == label else 0 for y in y_true_overall]
    y_true = np.array(y_true)

    y_pred = [1 if y == label else 0 for y in y_pred_overall]
    y_pred = np.array(y_pred)

    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    print(label, f1)
    list_scores.append(f1)

    support = len(data_merge[data_merge["Label"]==id2labels[label]])
    print(f"Support: {support}")
    print("")

print("Macro F1: ", np.mean(list_scores))
print("Weighted F1: ", np.average(list_scores, weights=[len(data_merge[data_merge["Label"]==label]) for label in id2labels.values()]))
