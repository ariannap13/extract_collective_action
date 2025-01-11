import os
import torch

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

torch.cuda.empty_cache()

prompt_v = 6

dpo = False
if dpo:
    dpo_tag_adapter = "dpo_results/"
    dpo_tag_output = "dpo_"

# Update the path accordingly
adapter_dir = f'../models/llama3_finetuned/{dpo_tag_adapter}simplified_v{prompt_v}_prompt_synthetic_more'
output_dir = f'../models/llama3_finetuned/merged_peft/{dpo_tag_output}simplified_v{prompt_v}_prompt_synthetic_more'

# set seed
torch.manual_seed(42)
model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)