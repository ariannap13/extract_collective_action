import pandas as pd
import bitsandbytes as bnb
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments)
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_dataset = "annotated_only" # "annotated_only" or "extended"
add_synthetic = True
if add_synthetic:
    synthetic_tag = "_synthetic"
else:
    synthetic_tag = ""
balanced = "balanced" 
layered = True
if layered:
    layered_tag = "_layered"
else:
    layered_tag = ""
add_more_synthetic = True
if add_more_synthetic:
    synthetic_tag_more = "_more"
else:
    synthetic_tag_more = ""
version_n = 6
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
if training_dataset == "annotated_only":
    perc = 100
else:
    perc = 95

prompt_v = 6

labels2ids = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3}
ids2labels = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution"}

# Define the collective action dimensions
## Llama3 definitions revised
def_dimension_v6 = {'Problem-Solution': "The comment highlights an issue and possibly suggests a way to fix it, often naming those responsible.",
                    'Call-to-Action': "The comment asks readers to take part in a specific activity, effort, or movement.",
                    'Intention': "The commenter shares their own desire to do something or be involved in solving a particular issue.",
                    'Execution': "The commenter is describing their personal experience taking direct actions towards a common goal."}

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

# remove rows with "None" label (not useful for training in the layered approach) 
data = data[data["Label"] != "None"]

## Shuffle the DataFrame
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

## Split the DataFrame
X_train, X_eval = train_test_split(data, test_size=0.1, random_state=42)

# Prompt definition 

## Define the prompt generation functions
def generate_prompt4(data_point):
    return f"""
            You have the following knowledge about collective action dimensions that can be expressed in social media comments: {dim_def}. Classify the following social media comment into one of the dimensions within the list {dim_def.keys()}, and return the answer as the corresponding collective action dimension label.
text: {data_point["text"]}
label: {data_point["majority_vote"]}""".strip()

def generate_prompt6(data_point):
    return f"""
            You have the following knowledge about levels of participation in collective action that can be expressed in social media comments: {dim_def}. 
            
            ### Definitions and Criteria:
            **Collective Action Problem:** A present issue caused by human actions or decisions that affects a group and can be addressed through individual or collective efforts.

            **Participation in collective action**: A comment must clearly reference a collective action problem, social movement, or activism by meeting at least one of the levels in the list {dim_def.keys()}.

            Classify the following social media comment into one of the levels within the list {list(dim_def.keys())}. 

            ### Example of correct output format:
            text: xyz
            label: None
            
            Return the answer as the corresponding participation in collective action level label.

            text: {data_point["text"]}
            label: {data_point["majority_vote"]}""".strip()


## Generate prompts for training and evaluation data
if prompt_v == 4:
    X_train.loc[:,'text'] = X_train.apply(generate_prompt4, axis=1)
    X_eval.loc[:,'text'] = X_eval.apply(generate_prompt4, axis=1)
elif prompt_v == 6:
    X_train.loc[:,'text'] = X_train.apply(generate_prompt6, axis=1)
    X_eval.loc[:,'text'] = X_eval.apply(generate_prompt6, axis=1)

# Prepare datasets and load model
train_data = Dataset.from_pandas(X_train[["text"]])
eval_data = Dataset.from_pandas(X_eval[["text"]])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# set seed
torch.manual_seed(42)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

tokenizer.pad_token_id = tokenizer.eos_token_id


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)

output_dir=f"../models/llama3_finetuned/{training_dataset}_v{version_n}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

training_arguments = TrainingArguments(
    output_dir=output_dir,                    # directory to save and repository id
    num_train_epochs=20,                       # number of training epochs
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    logging_steps=1,                         
    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
    group_by_length=False,
    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
    report_to="wandb",                  # report metrics to w&b
    eval_strategy="steps",              # save checkpoint every epoch
    eval_steps = 0.2,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False,
    dataset_kwargs={
    "add_special_tokens": False,
    "append_concat_token": False,
    }
)

trainer.train()

model.config.use_cache = True

# Save trained model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
