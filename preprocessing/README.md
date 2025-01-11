This folder contains some scripts that are useful for data pre-processing.

The scripts ``datasets_config.py`` and ``get_LLM_label_def.py`` are used to re-write label definitions.
 
``train_test_split.py`` performs, for each subreddit, a split into training and testing sets from the original Reddit data. 
Then, ``create_test_tobeannotated.py`` and ``create_train_tobeannotated.py`` perform sampling from the full training and test sets to create subsets undergoing human annotation.
Finally, ``action_focused_sents.py`` extracts the sentence containing the highest percentage of collective action words, the sentence before and the one after for all comments. This is used for the creation of training and test sets.