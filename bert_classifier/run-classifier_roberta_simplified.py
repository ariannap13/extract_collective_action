from simpletransformers.classification import ClassificationModel

import pandas as pd
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
cuda_available = torch.cuda.is_available()

add_synthetic_data = False

if add_synthetic_data:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    torch.cuda.empty_cache()

balance_strategy = "_weights"

# Data preparation
label2id = {"Action": 0, 
            "Non-action": 1}

id2label = {v: k for k, v in label2id.items()}

annotated_data = pd.read_csv("../data/train_set.csv")

if add_synthetic_data:
    annotated_data = annotated_data[annotated_data["AugmentationType"].isin(["None", "Synthetic"])]
else:
    annotated_data = annotated_data[annotated_data["AugmentationType"] == "None"]

annotated_data["simplified_label"] = ["Non-action" if x=="None" else "Action" for x in annotated_data["SimplifiedLabel"]]

# tranform "label" column to numerical using label2id
annotated_data["labels"] = annotated_data["simplified_label"].map(label2id)

tot_samples = annotated_data.shape[0]

output_dir = f"../models/collectiveaction_roberta_simplified{synthetic_tag}{balance_strategy}"
num_epochs = 30
manual_seed = 0
train_batch_size = 16

model_args = {
    'num_train_epochs':num_epochs,
    'fp16': True,
    "use_early_stopping": True,
    "output_dir": output_dir,
    "overwrite_output_dir": True,
    "manual_seed": manual_seed,
    "save_eval_checkpoints": False,
    "save_steps": -1,
    "train_batch_size": train_batch_size,
    "save_model_every_epoch": False,
    "learning_rate": 4e-5,
}

annotated_data = annotated_data[["ActionFocusedText", "labels"]]

if balance_strategy == "_weights":
    samples_class0 = annotated_data[annotated_data["labels"] == 0].shape[0]
    samples_class1 = annotated_data[annotated_data["labels"] == 1].shape[0]
    weights = tot_samples / (2 * np.array([samples_class0, samples_class1]))
    print(f"Class 0: {samples_class0}, Class 1: {samples_class1}, Weights: {weights}", flush=True)

num_labels = annotated_data['labels'].nunique()

# shuffle the dataset
data = annotated_data.sample(frac=1, random_state=manual_seed).reset_index(drop=True)

model = ClassificationModel('roberta', "../models/roberta_domain_finetuned",
    args=model_args,
    weight=list(weights),
    use_cuda=cuda_available,
    num_labels=num_labels,
)
 
model.train_model(data)