import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          pipeline)
from sklearn.metrics import (classification_report, 
                             confusion_matrix)

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
dpo = True
if dpo:
    dpo_tag = "dpo_"
else:
    dpo_tag = ""
version_n = 6
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
if training_dataset == "annotated_only":
    perc = 100
else:
    perc = 95

prompt_v = 6

model_dir = f'../models/llama3_finetuned/merged_peft/{dpo_tag}{training_dataset}_v{version_n}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}/final_merged_checkpoint' # f'./llama3_finetuned/merged_peft/{training_data}_v{version_n}/final_merged_checkpoint' if SFT or f'./llama3_finetuned/dpo_results/{training_data}_v{version_n}/final_checkpoint' if DPO

labels2ids = {"Problem-Solution": 0, "Call-to-action": 1, "Intention": 2, "Execution": 3, "None": 4}
ids2labels = {0: "Problem-Solution", 1: "Call-to-action", 2: "Intention", 3: "Execution", 4: "None"}

# Define the collective action dimensions
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
            You have the following knowledge about collective action dimensions that can be expressed in social media comments: {dim_def}. Classify the following social media comment into one of the dimensions within the list {dim_def.keys()}, and return the answer as the corresponding collective action dimension label.
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
if prompt_v == 4:
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt4, axis=1), columns=["text"])
elif prompt_v == 6:
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt6, axis=1), columns=["text"])

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
    categories = list(labels2ids.keys())
    
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=20, 
                        temperature=0.1)
        
        result = pipe(prompt)

        answer = result[0]['generated_text'].split("label:")[-1].strip()
        answers.append(answer)
        
        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("None")
    
    return y_pred, answers

y_pred, answer = predict(X_test, model, tokenizer)

# Save the predictions
predictions = pd.DataFrame({"CommentID": data["CommentID"], "text": X_test["text"], "y_true": y_true, "y_pred": y_pred, "full_answer": answer})
predictions.to_csv(f"../data/predictions/predictions_finetune_{dpo_tag}v{version_n}_{training_dataset}_prompt_v{prompt_v}_p{perc}{synthetic_tag}_{balanced}{synthetic_tag_more}{layered_tag}.csv", index=False)

# Evaluate the model
def evaluate(y_true, y_pred):
    labels = list(labels2ids.keys())
    mapping = {label: idx for idx, label in enumerate(labels)}
    
    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data
    
    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels, labels=list(range(len(labels))))
    print('\nClassification Report:', flush=True)
    print(class_report, flush=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
    print('\nConfusion Matrix:', flush=True)
    print(conf_matrix, flush=True)

evaluate(y_true, y_pred)