import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

test = False

if test:
    target_df = pd.read_csv("../data/test_set.csv")
    output_file = "../data/test_set.csv"
else:
    target_df = pd.read_csv("../data/train_set.csv")
    output_file = "../data/train_set.csv"

action_dic = pd.read_csv("../data/collective_action_dic.csv", names=['action']) # this file is not shared, from Smith et al. (2018)

def match_word(word, word_list):
    '''
    Function to match word with list of words.

    Args:
    word (str): input word to be matched
    word_list (list): list of words to match with

    Returns:
    boolean, match (exact of widlcard) or not
    '''
    count = 0
    for w in word_list:
        if word.endswith("*"):
            if w.startswith(word[:-1]):
                count += 1
        else:
            if word == w:
                count += 1
    return count

def get_action_sents(text):
    """
    Function to get frequency of actions in text.

    Args:
    text (str): input text.

    Returns:
    Text centered on action-infused sentences.
    """
    sents = sent_tokenize(text)

    list_sents = []
    for sent in sents:
        action_count = 0
        word_list_sent = word_tokenize(sent)
        for word in action_dic['action']:
            action_count += match_word(word, word_list_sent)
        list_sents.append([sent, action_count/len(word_list_sent)])

    # max action-infused sentence
    max_action_sent = max(list_sents, key=lambda x: x[1])
    # take that sentence, the one before and the one after and concatenate as text
    max_action_sent_idx = list_sents.index(max_action_sent)
    if len(list_sents) == 1:
        text = max_action_sent[0]
    elif max_action_sent_idx == 0:
        text = " ".join([max_action_sent[0], list_sents[max_action_sent_idx+1][0]])
    elif max_action_sent_idx == len(list_sents)-1:
        text = " ".join([list_sents[max_action_sent_idx-1][0], max_action_sent[0]])
    else:
        text = " ".join([list_sents[max_action_sent_idx-1][0], max_action_sent[0], list_sents[max_action_sent_idx+1][0]])
    
    return text

list_texts_tosave = []
for i, row in tqdm(target_df.iterrows()):
    text = row['OriginalText']
    action_count = 0
    word_list_sent = word_tokenize(text)
    for word in action_dic['action']:
        action_count += match_word(word, word_list_sent)
    if action_count < 2:
        continue
    action_text = get_action_sents(text)
    if test:
        list_texts_tosave.append(row['CommentID'], row['Subreddit'], row['OriginalText'], action_text, row['Label'])
    else:
        list_texts_tosave.append(row['CommentID'], row['Subreddit'], row['OriginalText'], action_text, row['Label'], row["AugmentationType"])

# save to csv
if test:
    df = pd.DataFrame(list_texts_tosave, columns=['CommentID', 'Subreddit', 'OriginalText', 'ActionFocusedText', 'Label'])
else:
    df = pd.DataFrame(list_texts_tosave, columns=['CommentID', 'Subreddit', 'OriginalText', 'ActionFocusedText', 'Label', 'AugmentationType'])

df.to_csv(output_file, index=False)
    