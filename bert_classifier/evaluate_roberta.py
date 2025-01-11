import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

add_synthetic_data = False

if add_synthetic_data:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""

balance_strategy = "_weights"

id2labels = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution", 4: "None"}
labels2id = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3, "None": 4}

# Open test data and clean annotations
data = pd.read_csv("../data/test_set.csv")

test_data = data.copy()

test_data["labels"] = [labels2id[label] for label in test_data["Label"]]

test_data = test_data.sort_values(by="CommentID")

test_data_idx = list(test_data["CommentID"].values)

test_pred = pd.read_csv(f"../data/predictions/predictions_roberta{synthetic_tag}{balance_strategy}.csv")
test_pred = test_pred[test_pred["CommentID"].isin(test_data_idx)]
test_pred = test_pred.sort_values(by="CommentID")
threshold = 0.5

y_true = test_data["labels"].values
y_pred = test_pred["predictions"].values

# Generate classification report
class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=["Problem-Solution", "Call-to-action", "Intention", "Execution", "None"], labels=[0,1,2,3,4])
print('\nClassification Report:', flush=True)
print(class_report, flush=True)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1,2,3,4])
print('\nConfusion Matrix:', flush=True)
print(conf_matrix, flush=True)