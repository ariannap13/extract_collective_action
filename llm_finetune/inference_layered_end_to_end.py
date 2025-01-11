import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, 
                             confusion_matrix)

training_dataset = "extended" # "annotated_only" or "extended"
add_synthetic = True
if add_synthetic:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""
add_more_synthetic = False
if add_more_synthetic:
    synthetic_tag_more = "_more"
else:
    synthetic_tag_more = ""
balanced = "balanced" 
layered = True
if layered:
    layered_tag = "_layered"
else:
    layered_tag = ""
dpo = True
if dpo:
    dpo_tag = "dpo_"
else:
    dpo_tag = ""
version_n = 6
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
if training_dataset == "annotated_only":
    perc = 100
else:
    perc = 95

prompt_v = 6

labels2ids = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3, "None": 4}
ids2labels = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution", 4: "None"}


# Open first layer predictions
predicted_data = pd.read_csv("../data/predictions/predictions_roberta_simplified_synthetic_weights.csv")

# open test set
data = pd.read_csv("../data/test_set.csv")
data["text"] = data["ActionFocusedText"]

data = data[data["CommentID"].isin(predicted_data["CommentID"])]

# sort both dataframes by id
data = data.sort_values(by="CommentID")
predicted_data = predicted_data.sort_values(by="CommentID")

# merge both dataframes
data_merge = pd.merge(data, predicted_data, on="CommentID")

# Open second layer predictions
data_second = pd.read_csv(f"../data/predictions/predictions_finetune_{dpo_tag}v{version_n}_{training_dataset}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}.csv")

i=0
final_pred = []
for idx, row in data_merge.iterrows():
    id = row["CommentID"]
    if row["predictions"] == 1:
        final_pred.append("None")
    else:
        final_pred.append(data_second[data_second["CommentID"] == id]["y_pred"].values[0])

data["final_pred"] = final_pred

y_true = data["Label"].values
y_pred = data["final_pred"].values

# Evaluate the model
def evaluate(y_true, y_pred):
    labels = list(labels2ids.keys())
    mapping = {label: idx for idx, label in enumerate(labels)}
    
    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data
    
    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels, labels=list(range(len(labels))))
    print('\nClassification Report:', flush=True)
    print(class_report, flush=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
    print('\nConfusion Matrix:', flush=True)
    print(conf_matrix, flush=True)

evaluate(y_true, y_pred)