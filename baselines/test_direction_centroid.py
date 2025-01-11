import pickle
import pandas as pd
import numpy as np

simplified = False
layered = False

if layered:
    layered_tag = "_layered"
else:
    layered_tag = ""

# Functions
def project_emb(emb, direction):
    """
    Project the embeddings onto the direction
    """

    emb = emb / np.linalg.norm(emb)
    direction = direction / np.linalg.norm(direction)
    score = np.dot(emb, direction.T) 

    return score

## data
df_filepath_training = f'../data/shuffled_training_None_df.pkl' # obtained from ../data_augmentation/reddit_extension/sbert_embeddings.py
training_df = pd.read_pickle(df_filepath_training)

if layered:
    training_df = training_df[training_df["Label"] != "None"]

text_training= training_df["ActionFocusedText"].tolist()

## embeddings
with open(f'../data/training_None_embeddings.pkl', 'rb') as f:
    training_embeddings = pickle.load(f)

# filter training embeddings based on the training_df
training_embeddings = training_embeddings[training_df.index]

# reset index
training_df = training_df.reset_index(drop=True)

# Test data 
df_filepath_test = f'../data/shuffled_test_df.pkl' # obtained from ../data_augmentation/reddit_extension/sbert_embeddings.py
test_df = pd.read_pickle(df_filepath_test)
test_data_ann = pd.read_csv("../data/test_set.csv")
test_df["text"] = test_df["ActionFocusedText"]
test_df = test_df[test_df["CommentID"].isin(test_data_ann["CommentID"])]

text_test= test_df["text"].tolist()

if layered:
    predicted_data = pd.read_csv("../data/predictions/predictions_roberta_simplified_synthetic_weights.csv") 

    ## Filter the data to only include the predicted action comments
    predicted_data = predicted_data[predicted_data["predictions"] == 0]

    test_df = test_df[test_df["CommentID"].isin(predicted_data["CommentID"])]

## embeddings
with open(f'../data/test_embeddings.pkl', 'rb') as f:
    test_embeddings = pickle.load(f)

# filter test embeddings based on the test_df
test_embeddings = test_embeddings[test_df.index]

# reset index
test_df = test_df.reset_index(drop=True)

## project
if simplified:
    training_df["simplified_label"] = ["Non-action" if x=="None" else "Action" for x in training_df["SimplifiedLabel"]]
    for label in training_df["simplified_label"].unique():
        print(f"Label: {label}")
        label_emb_train = training_embeddings[training_df[training_df["simplified_label"]==label].index]
        direction = np.mean(label_emb_train, axis=0)
        direction /= np.linalg.norm(direction)
        
        scores = [project_emb(emb, direction) for emb in test_embeddings]

        ### save scores_action
        with open(f'../data/test_scores_{label}.pkl', 'wb') as f:
            pickle.dump(scores, f)
else:
    for label in training_df["Label"].unique():
        print(f"Label: {label}")
        label_emb_train = training_embeddings[training_df[training_df["Label"]==label].index]
        direction = np.mean(label_emb_train, axis=0)
        
        scores = [project_emb(emb, direction) for emb in test_embeddings]

        ### save scores_action
        with open(f'../data/test_scores_{label}{layered_tag}.pkl', 'wb') as f:
            pickle.dump(scores, f)