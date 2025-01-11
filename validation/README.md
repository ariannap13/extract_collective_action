This folder contains the code to perform the validation of the proposed approach, by comparing it to topic modeling and stance detection, and by determining the impact of the use of the dictionary of collective action to select relevant samples on the performance on the test set. Moreover, it contains a classification analysis performed on a dataset different from Reddit: the set of transcriptions from the UK parliamentary debates.

# Topic Modeling
The notebook `topic_modeling.ipynb` performs the validation comparison with topic modeling. Note that predictions of participation in collective action on the dataset used for this specific validation can be obtained by running `../bert_classifier/inference-classifier_roberta.py` and selecting `topic` as dataset of reference.

# Stance Dectection
The notebook `stance_detection.ipynb` performs the validation comparison with stance detection. Note that predictions of participation in collective action on the dataset used for this specific validation can be obtained by running `../bert_classifier/inference-classifier_roberta.py` and selecting `GWSD` as dataset of reference.

# Action Keywords
The notebook `keywords_relevance.ipynb` is used to evaluate the impact of using the action words dictionary when selecting relevant samples for the training and test sets. 
Four different configurations of the test set are evaluated: (i) removal of action words, (ii) substitution of action words, (iii) removal of random words, and (iv) substitution of random words.
Note that predictions of participation in collective action on the dataset used for this specific validation can be obtained by running `../bert_classifier/inference-classifier_roberta.py` and selecting `no_action`, `sub_action`, `no_random`, or `sub_random` as dataset of reference.

# Classifying parliamentary debates
The notebook ``analyze_debates.ipynb`` contains an analysis of the classification of participation in collective action of UK parliamentary debates. Note that predictions of participation in collective action on the dataset used for this specific validation can be obtained by running `../bert_classifier/inference-classifier_roberta.py` and selecting `debates` as dataset of reference.

