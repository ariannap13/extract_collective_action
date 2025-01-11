This folder contains the scripts to perform LLaMa3 finetuning in terms of SFT and DPO.

# SFT
The script `finetune_generation_simplified.py` performs the finetuning of LLaMa3 for the binary task, in different configurations of the training set. Similarly, `finetune_generation.py` performs the finetuning for the multi-class task applied directly and `finetune_generation_layered.py` for the multi-class task as a second step of the layered approach.

When the fine-tuning is over, `merge_peft_adapter_simplified.py` and `merge_peft_adapter.py` are used to merge the adapters.

# DPO
The script `dpo_finetune_simplified.py` performs the finetuning of LLaMa3 for the binary task, in different configurations of the training set. Similarly, `dpo_finetune.py` performs the finetuning for the multi-class task applied directly and `dpo_finetune_layered.py` for the multi-class task as a second step of the layered approach.

When the fine-tuning is over, `merge_peft_adapter_simplified.py` and `merge_peft_adapter.py` are used to merge the adapters.

# Inference
The script `inference_finetune_simplified.py` is used to run the inference on the test set and print the evaluation performance for the binary task, `inference_finetune.py` for the multi-class task applied directly and `inference_finetune_layered.py` for the multi-class task as a second step of the multi-task layered approach. Finally, `inference_layered_end_to_end.py` computes and prints the performance metrics for the layered approach.