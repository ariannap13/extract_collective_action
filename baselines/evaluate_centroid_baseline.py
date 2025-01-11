import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import warnings

warnings.filterwarnings("ignore")

simplified = False
layered = False

if layered:
    layered_tag = "_layered"
else:
    layered_tag = ""

# Open test data and clean annotations
test_data = pd.read_pickle('../data/shuffled_test_df.pkl') # file obtained from ../data_augmentation/reddit_extension/sbert_embeddings.py

# Open annotated test data
test_data_ann = pd.read_csv("../data/test_set.csv")

test_data = test_data[test_data["CommentID"].isin(test_data_ann["CommentID"])]

test_data["simplified_label"] = ["Non-action" if label == "None" else "Action" for label in test_data["Label"]]

if layered:
    predicted_data = pd.read_csv("../data/predictions/predictions_roberta_simplified_synthetic_weights.csv") # not shared, contact the authors for access

    ## Filter the data to only include the predicted action comments
    predicted_data = predicted_data[predicted_data["predictions"] == 0]

    test_data = test_data[test_data["CommentID"].isin(predicted_data["CommentID"])]

if simplified:
    true_labels = test_data["simplified_label"].values

    list_performance = []
    # open prediction scores
    for label in test_data["simplified_label"].unique():
        if label == "Action":
            print(f"Label: {label}")
            with open(f'../data/test_scores_{label}.pkl', 'rb') as f:
                scores = pickle.load(f)
            # re-scale scores to 0-1 range with min max scaling
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            test_pred = [label if score >= 0.5 else "None" for score in scores]

            # modify true labels for the label to be evaluated
            true_labels = [label if true_label == label else "None" for true_label in true_labels]

            # Generate classification report
            class_report = classification_report(y_true=true_labels, y_pred=test_pred, target_names=["Action", "None"], labels=["Action", "None"])
            print('\nClassification Report:', flush=True)
            print(class_report, flush=True)
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_true=true_labels, y_pred=test_pred, labels=["Action", "None"])
            print('\nConfusion Matrix:', flush=True)
            print(conf_matrix, flush=True)
        
else:
    true_labels = test_data["Label"].values

    # open prediction scores
    list_performance = []
    for label in test_data["Label"].unique():
        if layered and label == "None":
            continue
        print(f"Label: {label}")
        with open(f'../data/test_scores_{label}{layered_tag}.pkl', 'rb') as f:
            scores = pickle.load(f)

        list_performance.append([x[0][0] for x in scores])
    

    # transpose list_performance
    list_performance = list(map(list, zip(*list_performance)))

    if layered:
        df_scores = pd.DataFrame(list_performance, columns=[label for label in test_data["Label"].unique() if label != "None"])
    else:
        df_scores = pd.DataFrame(list_performance, columns=test_data["Label"].unique())

    pred_labels = list(df_scores.idxmax(axis=1).values)
    true_labels = test_data["Label"].values

    df_scores["CommentID"] = test_data["CommentID"].values
    df_scores["predictions"] = pred_labels

    # export predictions
    df_scores.to_csv(f"../data/predictions/predictions_centroid{layered_tag}.csv", index=False)

    # Generate classification report
    if layered:
        class_report = classification_report(y_true=true_labels, y_pred=pred_labels, target_names=[label for label in test_data["Label"].unique() if label != "None"], labels=[label for label in test_data["Label"].unique() if label != "None"])
    else:
        class_report = classification_report(y_true=true_labels, y_pred=pred_labels, target_names=test_data["Label"].unique(), labels=test_data["Label"].unique())

    print('\nClassification Report:', flush=True)
    print(class_report, flush=True)

    # Generate confusion matrix
    if layered:
        conf_matrix = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=[label for label in test_data["Label"].unique() if label != "None"])
    else:
        conf_matrix = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=test_data["Label"].unique())

    print('\nConfusion Matrix:', flush=True)
    print(conf_matrix, flush=True)