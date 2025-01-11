import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig,  
                          pipeline)
from sklearn.metrics import (classification_report, 
                             confusion_matrix)

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

dpo = True

if dpo:
    model_tag = "dpo_"
else:
    model_tag = ""

prompt_v = 6

model_dir = f"../models/llama3_finetuned/merged_peft/{model_tag}simplified_v{prompt_v}_prompt_synthetic_more/final_merged_checkpoint"

# Data preparation
data = pd.read_csv("../data/test_set.csv")
data["text"] = data["ActionFocusedText"]

# change "Non-action" to "None" in the simplified_label column
data["simplified_label"] = ["None" if x=="None" else "Action" for x in data["SimplifiedLabel"]]
X_test = data.copy()

# Prompt definition 
def generate_test_prompt4_test(data_point):
    return f""""Classify whether the social media comment expresses collective action ("1") or not ("0").

    A comment is considered to express collective action if fits in any of the following descriptions: 
    * The comment highlights an issue and suggests a way to fix it, often naming those responsible.
    * The comment asks readers to take part in a specific activity, effort, or movement.
    * The commenter shares their own desire to do something or be involved in solving a particular issue.
    * The commenter is describing their personal experience taking direct actions towards a common goal.
    
    Return the label "1" or "0" based on the classification.

    Comment: {data_point["text"]}
    Label: """.strip()

def generate_test_prompt6_test(data_point):
    return f"""
    Classify the following social media comment as either "1" (expressing participation in collective action) or "0" (not expressing participation in collective action).

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

    Comment: "{data_point['text']}"
    Label: """.strip()

## Generate test prompts and extract true labels
y_true = X_test.loc[:,'simplified_label']
y_true = [1 if label == "Action" else 0 for label in y_true]
if prompt_v==4:
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt4_test, axis=1), columns=["text"])
elif prompt_v==6:
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt6_test, axis=1), columns=["text"])

# Prepare datasets and load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_dir)

tokenizer.pad_token_id = tokenizer.eos_token_id

# Define prediction 
def predict(test, model, tokenizer):
    y_pred = []
    answers = []
    
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=20, 
                        temperature=0.1)
        
        result = pipe(prompt)
        answer = result[0]['generated_text']
        answers.append(answer)

        # Look for first number following the "Label:" string (regex)

        if prompt_v==6:
            answer = answer.split("based on the classification.")[1]
            
        match = re.search(r'Label: (\d+)', answer)

        if match:
            y_pred.append(int(match.group(1)))
        else:
            y_pred.append(0)
        
    return y_pred, answers


y_pred, answer = predict(X_test, model, tokenizer)

# Save the predictions
predictions = pd.DataFrame({"CommentID": data["CommentID"], "text": X_test["text"], "y_true": y_true, "y_pred": y_pred, "full_answer": answer})
predictions.to_csv(f"../data/predictions/predictions_finetune_{model_tag}simplified_v{prompt_v}_prompt_synthetic_more.csv", index=False)

# Evaluate the model
def evaluate(y_true, y_pred):

    y_true = np.array([int(x) for x in y_true])
    y_pred = np.array([int(x) for x in y_pred])
    
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=["None", "Action"], labels=[0,1])
    print('\nClassification Report:', flush=True)
    print(class_report, flush=True)
    

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1])
    print('\nConfusion Matrix:', flush=True)
    print(conf_matrix, flush=True)

evaluate(y_true, y_pred)