import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import pandas as pd
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split

training_dataset = "extended" # "annotated_only" or "extended"
add_synthetic = True
if add_synthetic:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""
add_more_synthetic = False
if add_more_synthetic:
    synthetic_tag_more = "_more"
else:
    synthetic_tag_more = ""
balanced = "balanced" 
layered = False
if layered:
    layered_tag = "_layered"
else:
    layered_tag = ""
version_n = 6
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
if training_dataset == "annotated_only":
    perc = 100
else:
    perc = 95
n_examples_dpo = 2

prompt_v = 6
model_name = f'../models/llama3_finetuned/merged_peft/{training_dataset}_v{version_n}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}/final_merged_checkpoint'

# Define the collective action dimensions
## Llama3 definitions revised
def_dimension_v6 = {'Problem-Solution': "The comment highlights an issue and possibly suggests a way to fix it, often naming those responsible.",
                    'Call-to-Action': "The comment asks readers to take part in a specific activity, effort, or movement.",
                    'Intention': "The commenter shares their own desire to do something or be involved in solving a particular issue.",
                    'Execution': "The commenter is describing their personal experience taking direct actions towards a common goal.",
                    'None': "A comment doesn't fit into one of these categories; its purpose isn't clear or relevant to collective action."}

dim_def = def_dimension_v6

# Data preparation
if training_dataset == "annotated_only":
    data = pd.read_csv("../data/train_set.csv")
    data = data[data["AugmentationType"]=="None"]

elif training_dataset == "extended":
    data = pd.read_csv("../data/train_set.csv")
    data = data[data["AugmentationType"].isin(["None", "RedditExtension"])]    
    
if add_synthetic:
    data = data[(data["AugmentationType"].isin(["None", "Synthetic"]))&(data["Label"].isin(["Intention", "Execution"]))]
    if add_more_synthetic:
        data = data[data["AugmentationType"].isin(["None", "Synthetic"])]

data["text"] = data["ActionFocusedText"]

# split data in train and eval
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)

# Train data
all_train_data = []
for n in range(n_examples_dpo):
    all_train_data.append(train_data)
train_data = pd.concat(all_train_data)
train_data = train_data.reset_index(drop=True)
train_data["idx"] = train_data.index
# for each example, randomly select a negative label
# set seed
random.seed(42)
train_data["response_bad"] = train_data["Label"].apply(
    lambda x: random.choice([label for label in def_dimension_v6.keys() if label != x])
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
# set seed
random.seed(42)
eval_data["response_bad"] = eval_data["Label"].apply(
    lambda x: random.choice([label for label in def_dimension_v6.keys() if label != x])
)
# to Dataset
eval_data = Dataset.from_pandas(eval_data)


# Append task to each sample (this is needed when combining datasets)
train_data = train_data.map(
    lambda example: {
        "idx": example["idx"],
        "text": example["text"],
        "response_good": example["Label"],
        "response_bad": example["response_bad"],
        "task": "action-dimensions",
    }
)

eval_data = eval_data.map(
    lambda example: {
        "idx": example["idx"],
        "text": example["text"],
        "response_good": example["Label"],
        "response_bad": example["response_bad"],
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
        "prompt": [f"""You have the following knowledge about collective action dimensions that can be expressed in social media comments: {dim_def}. Classify the following social media comment into one of the dimensions within the list {dim_def.keys()}, and return the answer as the corresponding collective action dimension label.
            text: ```{input}```
            label: """ for input in samples["text"]],
        "chosen": samples["response_good"],
        "rejected": samples["response_bad"],
    }

def return_prompt_and_responses6(samples):
    return {
        "prompt": [f"""
            You have the following knowledge about levels of participation in collective action that can be expressed in social media comments: {dim_def}. 
            
            ### Definitions and Criteria:
            **Collective Action Problem:** A present issue caused by human actions or decisions that affects a group and can be addressed through individual or collective efforts.

            **Participation in collective action**: A comment must clearly reference a collective action problem, social movement, or activism by meeting at least one of the levels in the list {dim_def.keys()}.

            Classify the following social media comment into one of the levels within the list {list(dim_def.keys())}. 

            ### Example of correct output format:
            text: xyz
            label: None
            
            Return the answer as the corresponding participation in collective action level label.

            text: ```{input}```
            label: """ for input in samples["text"]],
        "chosen": samples["response_good"],
        "rejected": samples["response_bad"],
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

output_dir = f"../models/llama3_finetuned/dpo_results/{training_dataset}_v{version_n}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}"

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