import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
import sys
import tqdm
import re
from typing import List
import pandas as pd
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

sys.path.append('./') 

from datasets_config import (
    AdditionalDimConfig
)

def parse_output(input_string: str) -> List[str]:
    input_list = input_string.split("\n")  # split the input by newline characters
    output_list = []
    for item in input_list:
        item = item.lstrip(
            "0123456789. "
        )  # remove enumeration and any leading whitespace
        # remove any leading whitespace
        item = item.lstrip(" ")
        # remove any leading punctuation (use a regular expression)
        item = re.sub(r"^[^a-zA-Z0-9]+", "", item)
        if item:  # skip empty items
            output_list.append(item)
    return output_list


dimension = "Intention" # Change this to the dimension you want to generate data for
def_dimension = {'Problem-Solution': "The comment highlights an issue and possibly suggests a way to fix it, often naming those responsible.",
                    'Call-to-Action': "The comment asks readers to take part in a specific activity, effort, or movement.",
                    'Intention': "The commenter shares their own desire to do something or be involved in solving a particular issue.",
                    'Execution': "The commenter is describing their personal experience taking direct actions towards a common goal.",
                    'None': "A comment doesn't fit into one of these categories; its purpose isn't clear or relevant to collective action."}


df_examples = pd.read_csv(f"../../data/train_set.csv")
df_examples = df_examples[df_examples["AnnotationType"]=="None"]

df_subset = df_examples[df_examples["Label"]==dimension]

class HuggingfaceChatTemplate:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.use_default_system_prompt = False

    def get_template_augmentation(self, system_prompt: str, task: str) -> str:
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nAnchor example: {text}\nGenerated data: """.format(
                    task=task,
                    text="{text}",
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False)

if __name__ == "__main__":

    all_outputs = []
    original = []

    for index, row in tqdm.tqdm(df_subset.iterrows(), total=df_subset.shape[0]):

        all_outputs.append(row["ActionFocusedText"])
        original.append(1)

        config = AdditionalDimConfig

        llm = InferenceClient(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            token="", # insert your token here
        )

        template = HuggingfaceChatTemplate(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        ).get_template_augmentation(
            system_prompt=config.augmentation_system_prompt,
            task=config.augmentation_task_prompt,
        )

        label = row["Label"]

        output = llm.text_generation(
            template.format(
                label=label,
                social_dimension_description=def_dimension[label],
                text=row["ActionFocusedText"],
            ),
            max_new_tokens=800,
            temperature=0.1,
            )

        # Parse output
        parsed_output = parse_output(output)

        all_outputs.extend(parsed_output[2:])
        original.extend([0]*len(parsed_output[2:]))

    # Save outputs to df and csv
    df_outputs = pd.DataFrame(all_outputs)
    df_outputs.columns = ["ActionFocusedText"]
    df_outputs["original"] = original

    df_outputs.to_csv(f"../../data/{dimension}_synthetic.csv", index=False)