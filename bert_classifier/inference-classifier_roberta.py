from simpletransformers.classification import ClassificationModel

import pandas as pd
import numpy as np
import torch
import logging
import random

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
cuda_available = torch.cuda.is_available()

simplified = True
layered = False
add_synthetic_data = True

if add_synthetic_data:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""

if simplified:
    simplified_tag = "_simplified"
else:
    simplified_tag = ""

if layered:
    layered_tag = "_layered"
else:
    layered_tag = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    torch.cuda.empty_cache()

balance_strategy = "_weights" 

dataset = "test" # "test" or "topic" or "GWSD" or "no_action" or "sub_action" or "no_random" or "sub_random" or "downstream" or "debates"

def remove_random_token(text):
    tokens = text.split()
    np.random.seed(42)
    tokens.pop(random.randint(0, len(tokens) - 1))
    return " ".join(tokens)

def sub_action_random_word(text, vocab):
    tokens = text.split()
    # substitute each action word with the random word
    for i, token in enumerate(tokens):
        if token in values_action:
            random_word = random.choice(list(vocab.keys())).replace("Ġ", "")
            tokens[i] = random_word
    return " ".join(tokens)

def sub_random_random_word(text, vocab, p=0.1):
    tokens = text.split()
    # substitute each action word with the random word
    for i, token in enumerate(tokens):
        if random.random() < p:
            random_word = random.choice(list(vocab.keys())).replace("Ġ", "")
            tokens[i] = random_word
    return " ".join(tokens)

def match_words(words, text):
    '''
    Function to match a list of words with the words in the input text.

    Args:
    words (list): list of input words to be matched
    text (str): input text to evaluate for matches

    Returns:
    list of matched words
    '''
    word_list = text.split()  # Split the text into words
    list_matched_words = []
    
    for word in words:
        for w in word_list:
            if word.endswith("*"):  # Wildcard match
                if w.startswith(word[:-1]):
                    list_matched_words.append(w)
            else:  # Exact match
                if word == w:
                    list_matched_words.append(w)
                    
    return list_matched_words

if dataset!="test":
    dataset_tag = f"_{dataset}"
else:
    dataset_tag = ""

if dataset == "downstream": 
    dataset_type = "climate_change" 
    dataset_tag += f"_{dataset_type}"

id2label = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution", 4: "None"}
label2id = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3, "None": 4}

if layered:
    ## Read the dataset - prediction from binary classifier
    predicted_data = pd.read_csv(f"../data/predictions/predictions_roberta_simplified_synthetic_weights{dataset_tag}.csv")

    ## Filter the data to only include the predicted action comments
    predicted_data = predicted_data[predicted_data["predictions"] == 0]

# Open test set
if dataset == "test" or dataset == "no_action" or dataset == "sub_action" or dataset == "no_random" or dataset == "sub_random":
    data = pd.read_csv("../data/test_set.csv")

    # change "Non-action" to "None" in the simplified_label column
    data["simplified_label"] = ["None" if x=="None" else "Action" for x in data["SimplifiedLabel"]]

    if layered:
        data = data[data["CommentID"].isin(predicted_data["CommentID"])]

    data["text"] = data["ActionFocusedText"]

    df_test = data.copy()

elif dataset == "topic":
    data = pd.read_csv("../data/df_shuffled_topic.csv") # not shared, contact the authors for access
    data["text"] = data["body"]
    data["CommentID"] = data["id"]

    df_test = data.copy()

elif dataset == "GWSD":
    data = pd.read_csv("../data/df_shuffled_GWSD.csv") # not shared, built from the GWSD dataset released by Luo et al. (2020)
    data["text"] = data["sentence"]
    data["CommentID"] = data["sent_id"]

    df_test = data.copy()

elif dataset == "downstream":
    if dataset_type == "climate_change":
        data = pd.read_csv(f"../data/comments_top2k_2018_climate_change.csv") # not shared, contact the authors for access

    data["CommentID"] = data["id"]
    df_test = data.copy()

elif dataset == "debates":
    data = pd.read_csv("../data/df_shuffled_debates.csv") # from the UK Parliament debates dataset
    data["text"] = data["speech"]
    data["CommentID"] = list(range(data.shape[0]))

    df_test = data.copy()

df_test = df_test.dropna(subset=["text"])

## Read dictionary of action terms from Smith et al., 2018
df_dict_action = pd.read_csv('../data/collective_action_dic.csv', header=None)
values_action = df_dict_action[0].tolist()  

model_args = {"use_multiprocessing": False}

model = ClassificationModel('roberta', f"../models/collectiveaction_roberta{simplified_tag}{synthetic_tag}{balance_strategy}{layered_tag}", use_cuda=cuda_available, args=model_args)
vocab = model.tokenizer.get_vocab()

# Preprocess test set
if dataset == "no_action":
    # remove values_action from text, use the match_words function to match words
    df_test["text"] = df_test["text"].apply(lambda x: ' '.join([word if word not in match_words(values_action, x) else "" for word in x.split()]))
elif dataset == "sub_action":
    # substitute values_action with a placeholder
    df_test["text"] = df_test["text"].apply(sub_action_random_word, vocab=vocab)
elif dataset == "no_random":
    # remove 1 random word from each comment
    df_test["text"] = df_test["text"].apply(remove_random_token)
elif dataset == "sub_random":
    df_test["text"] = df_test["text"].apply(sub_random_random_word, vocab=vocab, p=0.1)

predictions, raw_outputs = model.predict(df_test["text"].tolist())

# extract probabilities from raw outputs
raw_outputs = torch.nn.functional.softmax(torch.tensor(raw_outputs), dim=1).tolist()

# save predictions
df_test['predictions'] = predictions
df_test["class_probabilities"] = raw_outputs

df_test.to_csv(f"../data/predictions/predictions_roberta{simplified_tag}{synthetic_tag}{balance_strategy}{layered_tag}{dataset_tag}.csv", index=False)