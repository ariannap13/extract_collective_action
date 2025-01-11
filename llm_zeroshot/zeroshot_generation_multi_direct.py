import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          pipeline)
from sklearn.metrics import (classification_report, 
                             confusion_matrix)

version_n = 6
prompt_v = 6
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

labels2ids = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3, "None": 4}
ids2labels = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution", 4: "None"}

## Llama3 definitions revised
def_dimension_v6 = {'Problem-Solution': "The comment highlights an issue and possibly suggests a way to fix it, often naming those responsible.",
                    'Call-to-Action': "The comment asks readers to take part in a specific activity, effort, or movement.",
                    'Intention': "The commenter shares their own desire to do something or be involved in solving a particular issue.",
                    'Execution': "The commenter is describing their personal experience taking direct actions towards a common goal.",
                    'None': "A comment doesn't fit into one of these categories; its purpose isn't clear or relevant to collective action."}

dim_def = def_dimension_v6

# Data preparation

## Read the dataset
data = pd.read_csv("../data/test_set.csv")
data["text"] = data["ActionFocusedText"]

X_test = data.copy()

# Prompt definition 

## Define the prompt generation functions
def generate_test_prompt4(data_point):
    return f"""
            You have the following knowledge about collective action dimensions that can be expressed in social media comments: {dim_def}. Classify the following social media comment into one of the dimensions within the list {list(dim_def.keys())}, and return the answer as the corresponding collective action dimension label.
text: {data_point["text"]}
label: """.strip()

def generate_test_prompt6(data_point):
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
            label: """.strip()

## Generate test prompts and extract true labels
y_true = X_test.loc[:,'Label']
if prompt_v == 6:
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt6, axis=1), columns=["text"])
elif prompt_v == 4:
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt4, axis=1), columns=["text"])

# Prepare datasets and load model
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

# Define prediction 
def predict(test, model, tokenizer):
    y_pred = []
    categories = list(labels2ids.keys())
    
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=20, 
                        temperature=0.01)
        
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()
        
        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("None")
    
    return y_pred, answer

y_pred, answer = predict(X_test, model, tokenizer)

# Save the predictions
predictions = pd.DataFrame({"CommentID": data["CommentID"], "text": X_test["text"], "y_true": y_true, "y_pred": y_pred, "full_answer": answer})
predictions.to_csv(f"../data/predictions/predictions_zeroshot_defs_v{version_n}_promptv_{prompt_v}.csv", index=False)

# Evaluate the model
def evaluate(y_true, y_pred):
    labels = list(labels2ids.keys())
    mapping = {label: idx for idx, label in enumerate(labels)}
    
    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data
    
    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=list(labels2ids.keys()), labels=list(range(len(labels))))
    print('\nClassification Report:', flush=True)
    print(class_report, flush=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
    print('\nConfusion Matrix:', flush=True)
    print(conf_matrix, flush=True)
    
evaluate(y_true, y_pred)