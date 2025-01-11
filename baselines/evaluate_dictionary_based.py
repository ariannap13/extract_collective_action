import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

labels = ["Action", "Non-action"]

# Open test data and clean annotations
data = pd.read_csv("../data/test_set.csv")

data["simplified_label"] = ["None" if x=="None" else "Action" for x in data["SimplifiedLabel"]]

test_data = data.copy()

test_data["clust_annotation"] = test_data["simplified_label"]

test_data = test_data.sort_values(by="CommentID")

test_data_idx = list(test_data["CommentID"].values)

test_pred = pd.read_csv(f"../data/predictions/predictions_dictionary_based.csv")
test_pred = test_pred[test_pred["CommentID"].isin(test_data_idx)]
test_pred = test_pred.sort_values(by="CommentID")

# Compute ROC-AUC, F1, Precision, Recall overall
list_scores = []
for label in labels:
    if label!="Action":
        continue
    y_true = (test_data["clust_annotation"] == label)*1
    y_true = np.array(y_true.values)

    y_pred = np.array(test_pred["perc_action_words"].values)
    roc_auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # get threshold on the top-right corner
    idx = np.argmax(tpr - fpr)
    threshold = thresholds[idx]
    print("Label: ", label, "Threshold: ", threshold)
    y_pred = (y_pred > threshold)*1

    print(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=["Action", "None"], labels=[1, 0])
    print('\nClassification Report:', flush=True)
    print(class_report, flush=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[1,0])
    print('\nConfusion Matrix:', flush=True)
    print(conf_matrix, flush=True)
