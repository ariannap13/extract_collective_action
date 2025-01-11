import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
import sys

# append a new directory to sys.path
sys.path.append('./')

from datasets_config import (LabelsDef
)

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def_dimension = {"Problem-Solution": "the comment identifies a problem related to collective action, assigning blame to the responsibles or proposing a plan of action on how to address the problem.", 
                 "Call-to-action": "the comment attempts to convince others to join or support a cause (e.g. a 'call to arms').",
                 "Intention": "the comment expresses the author's willingness or interest to act.",
                 "Execution": "the comment directly reports the author's involvement in some form of action.",
                 "None": "none of the above dimensions."}


labels_dims = list(def_dimension.keys())

class HuggingfaceChatTemplate:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.use_default_system_prompt = False

    def get_template_classification(self, system_prompt: str, task: str) -> str:
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nText: {text}\nAnswer: """.format(
                    task=task,
                    text="{text}",
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False)

if __name__ == "__main__":

    llm = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token="", # insert your token here
    )

    config = LabelsDef
    dimdef = def_dimension

    template = HuggingfaceChatTemplate(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    ).get_template_classification(
        system_prompt=config.classification_system_prompt,
        task=config.classification_task_prompt,
    )

    output = llm.text_generation(
        template.format(
            labels_dims = labels_dims,
            dimdef=dimdef,
            text=""
        ),
        max_new_tokens=200,
        temperature=0.9,
        repetition_penalty=1.2,
        )
    
    print(output)