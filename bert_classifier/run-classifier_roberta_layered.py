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
label2id = {"Problem-Solution": 0, 
               "Call-to-action": 1, 
               "Intention": 2,
               "Execution": 3}

id2label = {v: k for k, v in label2id.items()}

annotated_data = pd.read_csv("../data/train_set.csv")

if add_synthetic_data:
    annotated_data = annotated_data[annotated_data["AugmentationType"].isin(["None", "Synthetic"])]
else:
    annotated_data = annotated_data[annotated_data["AugmentationType"] == "None"]

annotated_data = annotated_data[annotated_data["Label"]!="None"]

# tranform "label" column to numerical using label2id
annotated_data["labels"] = annotated_data["Label"].map(label2id)

tot_samples = annotated_data.shape[0]

output_dir = f"../models/collectiveaction_roberta{synthetic_tag}{balance_strategy}_layered"
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
    array_weights = []
    for label in label2id:
        sampled_class = annotated_data[annotated_data["labels"] == label2id[label]].shape[0]
        array_weights.append(sampled_class)
    array_weights = np.array(array_weights)
    weights = tot_samples / (2 * array_weights)
    print(f"Weights: {weights}", flush=True)

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