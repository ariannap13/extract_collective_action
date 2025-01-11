import pandas as pd

dataset = "test" 

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

# Open test set
if dataset == "test":
    data = pd.read_csv("../data/test_set.csv")

    data["simplified_label"] = ["None" if x=="None" else "Action" for x in data["SimplifiedLabel"]]
    df_test = data.copy()

df_test["labels"] = df_test["simplified_label"].apply(lambda x: 0 if x == "Action" else 1)

## Read dictionary of action terms (from Smith et al., 2018)
df_dict_action = pd.read_csv('../data/collective_action_dic.csv', header=None)
values_action = df_dict_action[0].tolist()  

df_test["perc_action_words"] = df_test["ActionFocusedText"].apply(lambda x: len(match_words(values_action, x))/len(x.split()) if len(x.split()) > 0 else 0)

df_test.to_csv(f"../data/predictions/predictions/predictions_dictionary_based.csv", index=False)