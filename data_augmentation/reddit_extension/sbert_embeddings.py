from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pickle
import numpy as np
import torch
from tqdm import tqdm

MAX_SENTENCE_LENGTH = 256
OVERLAP = 64

training = False
if training:
    augmentation_type = "RedditExtension" # "RedditExtension" or "Synthetic" or "None"
    tag = f"training_{augmentation_type}"
else:
    tag = "test"

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def create_windows(input_ids, attention_mask, window_size, step_size):
    # Create windows from input_ids
    windows_input = []
    windows_attention_mask = []
    sentence_indices = []

    for i in range(len(input_ids)):

        if len(input_ids[i]) <= window_size:
            # pad the input if necessary
            padding = [0] * (window_size - len(input_ids[i]))
            input_ids[i] = input_ids[i] + padding
            attention_mask[i] = attention_mask[i] + padding
            windows_input.append(input_ids[i])
            windows_attention_mask.append(attention_mask[i])
            sentence_indices.append(i)
            continue

        for j in range(0, len(input_ids[i]), step_size):
            window_input = input_ids[i][j:j + window_size]
            window_attention_mask = attention_mask[i][j:j + window_size]

            # Pad the window if necessary
            if len(window_input) <= window_size:
                padding = [0] * (window_size - len(window_input))
                window_input = window_input + padding
                window_attention_mask = window_attention_mask + padding
            
            windows_input.append(window_input)
            windows_attention_mask.append(window_attention_mask)
            sentence_indices.append(i)

    return list(zip(windows_input, windows_attention_mask, sentence_indices))

# Load data

## Read the dataset
if training:
    data = pd.read_csv("../../data/train_set.csv")
    data = data[data["AugmentationType"] == augmentation_type]
else:
    data = pd.read_csv("../../data/test_set.csv")

# Embeddings

## Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

## shuffle df 
df = data.sample(frac=1, random_state=0).reset_index(drop=True)

## save shuffled df
df_filepath = f'../../data/shuffled_{tag}_df.pkl'

with open(df_filepath, 'wb') as f:
    pickle.dump(df, f)

text = df['ActionFocusedText'].tolist()

## Tokenize sentences
encoded_input = tokenizer(text, truncation=False)

## Create windows from encoded input
windows_data = create_windows(encoded_input['input_ids'], encoded_input['attention_mask'], window_size=MAX_SENTENCE_LENGTH, step_size=OVERLAP)

## count how many windows per sentence
sentence_windows_count = {}
for _, _, sentence_index in windows_data:
    if sentence_index in sentence_windows_count:
        sentence_windows_count[sentence_index] += 1
    else:
        sentence_windows_count[sentence_index] = 1

## Compute embeddings for each window and mean pool for each sentence
sentence_embeddings = {}
for window, attention_mask, sentence_index in tqdm(windows_data):
    
    window_input_tensor = torch.tensor([window])
    attention_mask_tensor = torch.tensor([attention_mask])

    with torch.no_grad():
        model_output = model(input_ids=window_input_tensor, attention_mask=attention_mask_tensor)

    window_embedding = mean_pooling(model_output, attention_mask_tensor)

    if sentence_index in sentence_embeddings:
        sentence_embeddings[sentence_index] += window_embedding.detach().numpy()/sentence_windows_count[sentence_index]
    else:
        sentence_embeddings[sentence_index] = window_embedding.detach().numpy()/sentence_windows_count[sentence_index]

## Normalize aggregated embeddings
sentence_indices, aggregated_embeddings = zip(*sentence_embeddings.items())
aggregated_embeddings = np.array(aggregated_embeddings)

# Save the embeddings
avgpool_embeddings_filepath = f'../../data/{tag}_embeddings.pkl'

with open(avgpool_embeddings_filepath, 'wb') as f:
    pickle.dump(aggregated_embeddings, f)
