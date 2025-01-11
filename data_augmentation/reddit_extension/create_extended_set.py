import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

labels = ["Problem-Solution", "Call-to-action", "Intention", "Execution", "None"]

perc = 95
tag = "same_thread_similar_examples"

# Load embeddings and text of the whole dataset (with action)
df_filepath = f'../../data/shuffled_all_training_set_action_focused_df.pkl' # not shared, contact the authors for access
df_whole = pd.read_pickle(df_filepath)
df_whole = df_whole.dropna(subset=["text"])
df_whole_fulltext = pd.read_pickle(f'../../data/shuffled_all_training_set_df.pkl') # not shared, contact the authors for access
df_whole_fulltext = df_whole_fulltext.dropna(subset=["text"])
avgpool_embeddings_filepath = f'../../data/embeddings_sbert_all_training_set_action_focused.pkl' # obtained from sbert_embeddings.py run on the whole dataset

with open(avgpool_embeddings_filepath, 'rb') as f:
    aggregated_embeddings_whole = pickle.load(f)

## reset index
df_whole = df_whole.reset_index(drop=True)

# Load embeddings and text of the annotated training
df_filepath = f'../../data/shuffled_training_None.pkl' # output of sbert_embeddings.py. Note that sub_ids, needed for this augmentation, is not directly shared in train_set.csv. Contact the authors for access.
df_annot = pd.read_pickle(df_filepath)
df_annot = df_annot.dropna(subset=["ActionFocusedText"])
df_annot = df_annot.dropna(subset=["Subreddit"]).reset_index(drop=True)

avgpool_embeddings_filepath = f'../../data/training_None_embeddings.pkl' # output of sbert_embeddings.py.

with open(avgpool_embeddings_filepath, 'rb') as f:
    aggregated_embeddings_annot = pickle.load(f)

# Platform-informed selection
for subreddit in df_annot["Subreddit"].unique():
    mask = df_whole["subreddit"] == subreddit
    df_whole_sub = df_whole[mask]
    embeddings_whole = aggregated_embeddings_whole[mask]
    for label in labels:
        df_annot_sub = df_annot[(df_annot["Subreddit"] == subreddit) & (df_annot["Label"] == label)]

        if len(df_annot_sub) == 0:
            continue

        embeddings_annot = aggregated_embeddings_annot[(df_annot["Subreddit"] == subreddit) & (df_annot["Label"] == label)]
        embeddings_annot = embeddings_annot.reshape(embeddings_annot.shape[0], -1)

        ## Same thread
        threads_annot = df_annot_sub["link_id"].unique()
        mask_thread = df_whole_sub["link_id"].isin(threads_annot)
        df_whole_sub_same_thread = df_whole_sub[mask_thread]
        # remove duplicates of id between df_whole_sub_same_thread and df_annot_sub
        df_whole_sub_same_thread = df_whole_sub_same_thread[~df_whole_sub_same_thread["id"].isin(df_annot_sub["CommentID"])]
        same_thread_examples = df_whole_sub_same_thread["text"].tolist()

        ## Same thread, most similar
        embeddings_whole_same_thread = aggregated_embeddings_whole[df_whole_sub_same_thread.index]
        if len(embeddings_whole_same_thread) == 0:
            continue
        similarities = cosine_similarity(embeddings_annot, embeddings_whole_same_thread)
        # Remove elements that are too similar -- possible repeated entries or bots
        similarities[(similarities >= 0.95)] = 0
        # get 90th percentile of similarities
        min_sim_threshold = np.percentile(similarities, perc)
        print("Threshold: ", min_sim_threshold, flush=True)
        # Filter by threshold 
        similarities[(similarities < min_sim_threshold)] = 0
        similar_examples = []
        for i in range(similarities.shape[1]):
            if similarities[:, i].any() > 0:
                id_similar = df_whole_sub_same_thread.iloc[i]["id"]
                try:
                    full_text = df_whole_fulltext[df_whole_fulltext["id"] == id_similar]["text"].values[0]
                except:
                    full_text = "None"
                similar_examples.append([id_similar, full_text, df_whole_sub_same_thread.iloc[i]["text"], similarities[:, i].max()])
        # Save the list of similar examples
        with open(f'../../data/same_thread_similar_examples_{subreddit}_{label}_p{perc}.pkl', 'wb') as f:
            pickle.dump(similar_examples, f)

# Export single file containing all extensions
list_dfs = []
for label in labels:

    for file in os.listdir("../../data/"):
        if file.startswith(tag) and file.endswith(f"_{label}_p{perc}.pkl"):
            if len(file.split("_")) > 7:
                subreddit = file.split("_")[-3] + "_" + file.split("_")[-2]
            else:
                subreddit = file.split("_")[-3]
            # read pickle list
            with open(f"../../data/{file}", 'rb') as f:
                list_sim = pickle.load(f)
            list_ids = [x[0] for x in list_sim]
            list_fulltexts = [x[1] for x in list_sim]
            list_texts = [x[2] for x in list_sim]
            list_subreddits = [subreddit] * len(list_texts)
            list_labels = [label] * len(list_texts)
            list_original = [0] * len(list_texts)
            df = pd.DataFrame(list(zip(list_ids, list_fulltexts, list_texts, list_subreddits, list_labels, list_original)), columns=["id", "fulltext", "text", "subreddit", "label", "original"])
            list_dfs.append(df)

df = pd.concat(list_dfs)

# Save dataframe
df.to_pickle(f'../../data/training_set_extended_{tag}_p{perc}.pkl')
