This folder contains the code to train and evaluate the RoBERTa Base classifier.

# RoBERTa Base Fine-Tuning
The script ``finetune_roberta_domain.py`` is needed to fine-tune the RoBERTa Base model on the Reddit domain. The finetuned model is the starting point for the classification training step.

# Classification Model
Depending on the type of task, the following scripts are used to train the RoBERTa classification model:
* `run-classifier_roberta_simplified.py` for the binary task.
* `run-classifier_roberta.py` for the multi-class task applied directly.
* `run-classifier_roberta_layered.py`for the multi-class task considered as a second step of the layered approach.

# Evaluation
The script `inference-classifier_roberta.py` produces the predicted labels for the test set, while the scripts `evaluate_simplified_roberta.py`, `evaluate_roberta.py`,  and `evaluate_layered_end_to_end.py` describe the evaluation performance metrics for the binary task, the multi-class task performed directly and the multi-class layered task, respectively.
