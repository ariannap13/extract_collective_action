import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score)

labels = ["Problem-Solution", "Call-to-action", "Intention", "Execution", "None"]

labels2id = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3, "None": 4}
id2labels = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution", 4: "None"}

# Open first layer predictions
predicted_data = pd.read_csv("../data/predictions/predictions_roberta_simplified_synthetic_weights.csv") # not shared, contact the authors for access

# open test set
data = pd.read_csv("../data/test_set.csv")

data = data[data["CommentID"].isin(predicted_data["CommentID"])]

# sort both dataframes by id
data = data.sort_values(by="CommentID")
predicted_data = predicted_data.sort_values(by="CommentID")

# merge both dataframes
data_merge = pd.merge(data, predicted_data, on="CommentID")

# Open second layer predictions
data_second = pd.read_csv(f"../data/predictions/predictions_centroid_layered.csv")

final_pred = []
for idx, row in data_merge.iterrows():
    id = row["CommentID"]
    if row["predictions"] == 1:
        final_pred.append(4)
    else:
        final_pred.append(labels2id[data_second[data_second["CommentID"] == id]["predictions"].values[0]])

data["final_pred"] = final_pred

y_true_overall = [labels2id[label] for label in data["Label"].values]
y_pred_overall = data["final_pred"].values

# Compute ROC-AUC, F1, Precision, Recall overall
list_scores = []
for label in id2labels:
    y_true = [1 if y == label else 0 for y in y_true_overall]
    y_true = np.array(y_true)

    y_pred = [1 if y == label else 0 for y in y_pred_overall]
    y_pred = np.array(y_pred)

    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    print(id2labels[label], f1)
    list_scores.append(f1)

print("Macro F1: ", np.mean(list_scores))
print("Weighted F1: ", np.average(list_scores, weights=[len(data[data["Label"]==label]) for label in id2labels.values()]))
