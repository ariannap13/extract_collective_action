import os
import torch

torch.cuda.empty_cache()

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

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
dpo = False
if dpo:
    dpo_tag_adapter = "dpo_results/"
    dpo_tag_output = "dpo_"
else:
    dpo_tag_adapter = ""
    dpo_tag_output = "sft_"

version_n = 6
prompt_v = 6
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
if training_dataset == "annotated_only":
    perc = 100
else:
    perc = 95

# Update the path accordingly
adapter_dir = f"../models/llama3_finetuned/{dpo_tag_adapter}collectiveaction_{dpo_tag_output}{training_dataset}_v{version_n}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}"
output_dir = f"../models/llama3_finetuned/merged_peft/collectiveaction_{dpo_tag_output}{training_dataset}_v{version_n}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}"

# set seed
torch.manual_seed(42)
model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=torch.bfloat16, local_files_only=True)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
