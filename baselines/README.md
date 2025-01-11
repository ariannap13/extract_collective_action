This folder contains the code to train and apply baseline models for the classification of participation in collective action and its levels (binary and multi-class tasks).

# Dictionary-based classifier
In the binary task, one of the reference baselines considers the percentage of collective action words contained in each comment as predictor of its expressed participation in collective action. 
The script ``compare_dictionary_approach.py`` defines the percentages while ``evaluate_dictionary_based.py`` computes the performance metrics.

# Centroid classifier
Both in the binary and in the multi-class tasks, embedding and LLM-based NLP models are compared to a centroid classifier. The script `../data_augmentation/reddit_extension/sbert_embeddings.py` extracts sentence-bert embeddings for either the training or the test sets and `test_direction_centroid.py` defines the centroid for a given dimension and computes the cosine similarity between such a centroid and a test sample. To compute the performance on the test set, we then use `evaluate_centroid_baseline.py` for the binary task and the multi-class task performed directly, and `evaluate_centroid_layered_end_to_end.py` for the multi-class layered approach.
