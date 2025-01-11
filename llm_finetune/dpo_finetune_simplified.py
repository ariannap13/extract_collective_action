import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import pandas as pd
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split

n_examples_dpo = 1

model_name = f'../models/llama3_finetuned/merged_peft/simplified_v6_prompt_synthetic_more/final_merged_checkpoint'

prompt_v = 6

add_synthetic = True
if add_synthetic:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""
add_more_synthetic = True
if add_more_synthetic:
    synthetic_tag_more = "_more"
else:
    synthetic_tag_more = ""

# Data preparation
## Read the dataset
data = pd.read_csv("../data/train_set.csv")
data["text"] = data["ActionFocusedText"]

if add_synthetic:
    data = data[(data["AugmentationType"].isin(["None", "Synthetic"]))&(data["Label"].isin(["Intention", "Execution"]))]
    if add_more_synthetic:
        data = data[data["AugmentationType"].isin(["None", "Synthetic"])]
else:
    data =  data = data[data["AugmentationType"]=="None"]

data["simplified_label"] = ["Non-action" if x=="None" else "Action" for x in data["Label"]]

# change "Non-action" to "None" in the simplified_label column
data["simplified_label"] = data["simplified_label"].apply(lambda x: "None" if x == "Non-action" else x)
data["simplified_label"] = [1 if x == "Action" else 0 for x in data["simplified_label"]]


## Shuffle the DataFrame
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# split data in train and eval
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)

labels2ids = {"Action": 1, "None": 0}
ids2labels = {1: "Action", 0: "None"}

# Train data
all_train_data = []
for n in range(n_examples_dpo):
    all_train_data.append(train_data)
train_data = pd.concat(all_train_data)
train_data = train_data.reset_index(drop=True)
train_data["idx"] = train_data.index
# for each example, randomly select a negative label
train_data["response_bad"] = train_data["simplified_label"].apply(
    lambda x: random.choice([label for label in ids2labels.keys() if label != x])
)
# to Dataset
train_data = Dataset.from_pandas(train_data)

# Eval data
all_eval_data = []
for n in range(n_examples_dpo):
    all_eval_data.append(eval_data)
eval_data = pd.concat(all_eval_data)
eval_data = eval_data.reset_index(drop=True)
eval_data["idx"] = eval_data.index
# for each example, randomly select a negative label
eval_data["response_bad"] = eval_data["simplified_label"].apply(
    lambda x: random.choice([label for label in ids2labels.keys() if label != x])
)
# to Dataset
eval_data = Dataset.from_pandas(eval_data)


# Append task to each sample (this is needed when combining datasets)
train_data = train_data.map(
    lambda example: {
        "idx": example["idx"],
        "text": example["text"],
        "response_good": str(example["simplified_label"]),
        "response_bad": str(example["response_bad"]),
        "task": "action-dimensions",
    }
)

eval_data = eval_data.map(
    lambda example: {
        "idx": example["idx"],
        "text": example["text"],
        "response_good": str(example["simplified_label"]),
        "response_bad": str(example["response_bad"]),
        "task": "action-dimensions",
    }
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# set seed
torch.manual_seed(42)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# set seed
torch.manual_seed(42)

model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def return_prompt_and_responses4(samples):
    return {
        "prompt": [f"""Classify whether the social media comment expresses collective action ("1") or not ("0").

                    A comment is considered to express collective action if fits in any of the following descriptions: 
                    * The comment highlights an issue and suggests a way to fix it, often naming those responsible.
                    * The comment asks readers to take part in a specific activity, effort, or movement.
                    * The commenter shares their own desire to do something or be involved in solving a particular issue.
                    * The commenter is describing their personal experience taking direct actions towards a common goal.
                    
                    Return the label "1" or "0" based on the classification.
            Comment: ```{input}```
            Label: """ for input in samples["text"]],
        "chosen": [str(label) for label in samples["response_good"]],
        "rejected": [str(label) for label in samples["response_bad"]],
    }


def return_prompt_and_responses6(samples):
    return {
        "prompt": [f"""Classify the following social media comment as either "1" (expressing participation in collective action) or "0" (not expressing participation in collective action).

            ### Definitions and Criteria:
            **Collective Action Problem:** A present issue caused by human actions or decisions that affects a group and can be addressed through individual or collective efforts.

            **Participation in collective action**: A comment must clearly reference a collective action problem, social movement, or activism by meeting at least one of the following:
            1. The comment identifies the issue as a problem and optionally proposes solutions and/or assigns responsibility.
            2. The comment encourages others to take action or join a cause.
            3. The comment expresses personal intent to act or current involvement in activism.

            ### Labeling Instructions:
            - Label the comment as "1" if it expresses participation in collective action.
            - Label the comment as "0" if it does not express participation in collective action.

            ### Example of correct output
            Comment: "xyz"
            Label: 0

            Return the label "1" or "0" based on the classification.
                   
            Comment: ```{input}```
            Label: """ for input in samples["text"]],
        "chosen": [str(label) for label in samples["response_good"]],
        "rejected": [str(label) for label in samples["response_bad"]],
    }


train_data = train_data
eval_data = eval_data
original_columns = train_data.column_names

if prompt_v == 6:
    dataset_train = train_data.map(
        return_prompt_and_responses6,
        batched=True,
        remove_columns=original_columns
    )

    dataset_eval = eval_data.map(
        return_prompt_and_responses6,
        batched=True,
        remove_columns=original_columns
    )
elif prompt_v == 4:
    dataset_train = train_data.map(
        return_prompt_and_responses4,
        batched=True,
        remove_columns=original_columns
    )

    dataset_eval = eval_data.map(
        return_prompt_and_responses4,
        batched=True,
        remove_columns=original_columns
    )

output_dir = f"../models/llama3_finetuned/dpo_results/simplified_v{prompt_v}_prompt{synthetic_tag}{synthetic_tag_more}"

training_args = DPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=10, 
    save_steps= 0.1,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    remove_unused_columns=False,
    seed=42,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
    max_prompt_length=1024,
    max_length=512,
)


dpo_trainer.train()
dpo_trainer.save_model(output_dir)


output_dir = os.path.join(output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)