from simpletransformers.language_modeling import  LanguageModelingModel
import torch
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
cuda_available = torch.cuda.is_available()

trainfile='../data/full_train_action_focused.txt' # this file is not shared, contact the authors for access
outdir='../models/roberta_domain_finetuned/' # this directory is not shared, contact the authors for access

model_args = {
        'num_train_epochs':5,
        "use_early_stopping": True,
        "output_dir": outdir,
        "overwrite_output_dir": True,
        "manual_seed": 42,
        "save_eval_checkpoints": False,
        "save_steps": -1,
        "train_batch_size": 128,
        "n_gpu": 2,
}

model = LanguageModelingModel("roberta", "roberta-base", args=model_args,use_cuda=cuda_available)
model.train_model(trainfile)