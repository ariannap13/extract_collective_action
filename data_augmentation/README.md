This folder contains the scripts to perform two types of data augmentation:

1. Synthetic: by exploiting LLama 3 8B Instruct, starting from crowdworkers annotated data. 
    In ``./synthetic``, ``huggingface_prompting_augmentation.py`` is used to generate synthetic text samples.
2. Reddit Extension: starting from the full set of Reddit comments posted in activism-related communities (available upon request) and the crowdworkers annotated data, for each annotated sample we identified the most similar examples within the same thread and assigned the same label.
    In ``./reddit_extension``, there is ``sbert_embeddings.py`` used to extract sbert embeddings textual traces, and ``create_extended_set.py`` to generate the Reddit-informed dataset extension through embeddings similarity.

Note that, while working data (and embeddings) are not shared, the final result of the results of the data augmentation process are shared as part of the training set, in ``../data/train_set.csv`` with `AugmentationType` as `Synthetic` and `RedditExtension`, respectively.