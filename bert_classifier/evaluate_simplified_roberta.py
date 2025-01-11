import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

add_synthetic_data = False

if add_synthetic_data:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""

balance_strategy = "_weights"

# Open test data and clean annotations
data = pd.read_csv("../data/test_set.csv")

labels = ["Action", "None"]

# change "Non-action" to "None" in the simplified_label column
data["simplified_label"] = ["None" if x=="None" else "Action" for x in data["SimplifiedLabel"]]

test_data = data.copy()

test_data = test_data.sort_values(by="CommentID")

test_data_idx = list(test_data["CommentID"].values)

test_pred = pd.read_csv(f"../predictions/predictions_roberta_simplified{synthetic_tag}{balance_strategy}.csv")
test_pred = test_pred[test_pred["CommentID"].isin(test_data_idx)]
test_pred = test_pred.sort_values(by="CommentID")
threshold = 0.5

test_pred["labels"] = [1 if x=="None" else 0 for x in test_pred["simplified_label"]]

y_true = test_pred["labels"].values
y_pred = test_pred["predictions"].values

# Generate classification report
class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=["Action", "None"], labels=[0,1])
print('\nClassification Report:', flush=True)
print(class_report, flush=True)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1])
print('\nConfusion Matrix:', flush=True)
print(conf_matrix, flush=True)