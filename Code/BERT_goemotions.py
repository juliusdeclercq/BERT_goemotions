# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 18:53:19 2023

@author: Julius de Clercq
"""

#%%                 Imports
import numpy as np
import pandas as pd
from time import time as t
import warnings
import pickle
import os 
import pathlib as pl 
 
from datasets import Dataset, DatasetDict #, load_dataset
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

#%%                 Downloading the Goemotions data

def Goemotions_merge():
    """
    The Goemotions data is split into three .csv files. These files are merged into
    a single dataframe here.
    """
    df = pd.DataFrame()
    generic_filepath = os.path.join(pl.parent.parent, "Input/Goemotions data/goemotions")
    for i in [1,2,3]:
        try:
            data = pd.read_csv(f"{generic_filepath}_{i}.csv")
        except:     # If this fails, it means we already removed the very sizable datasets. 
            break
        df = df.append(data)
        del data        # Deleting unused data to save memory
    df.to_csv("Input/Goemotions")


# Goemotions_merge()

df = pd.read_csv("Input/Goemotions")


#%%            Cleaning the Goemotions data

# Also, I want to remove observations which are marked as very unclear. 
# This is because it shrinks the dataset so it speeds up computation, and the 
# examples are not very informative anyways, as they are marked as unclear. 
# This only removes ~3k out of 211k documents.
df = df[df['example_very_unclear'] != True]

# I want to remove unnecessary columns for making the dataset less 
# memory/computationally intensive.
columns_to_remove = ['Unnamed: 0',
                     'id', 
                     'author', 
                     'subreddit', 
                     'link_id', 
                     'parent_id', 
                     'created_utc', 
                     'rater_id', 
                     'example_very_unclear',
                     ]
# '__index_level_0__'
df = df.drop(columns=columns_to_remove)

# subsampling data to see if this resolves memory issues
# subsample_fraction = 0.1
# df = df.sample(frac=subsample_fraction, random_state=765432)

#%%             Train-test splitting
def traintestsplitting(df, val_ratio):
    """ 
    Matching the data format of the example from TDS by storing the dataframes 
    in a dictionary and saving them as a Dataset object from Huggingface.
    """
    # Split indices into training and validation sets
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_ratio,
        shuffle=True
    )
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_train.reset_index(drop=True, inplace=True)
    df_val .reset_index(drop=True, inplace=True)
    return df_train, df_val

val_ratio = 0.2  # Fraction of data for validation set.
df_train, df_val = traintestsplitting(df, val_ratio)
dataset = {'train': Dataset.from_pandas(df_train),
            'validation': Dataset.from_pandas(df_val)
            }
dataset = DatasetDict(dataset)

#%%             Label list and dictionaries
"""
Let's create a list that contains the labels, as well as 2 dictionaries that map labels to integers and back.
"""

labels = [label for label in dataset['train'].features.keys() if label not in ['text','__index_level_0__']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(labels)

label_df = {'labels': labels,
            'id2label': id2label,
            'label2id': label2id}

# Save the dictionary to a local file using pickle
with open("label_df.pkl", "wb") as f:
    pickle.dump(label_df, f)

#%%             Preprocessing and tokenization
""""
As models like BERT don't expect text as direct input, but rather `input_ids`, etc., we tokenize the text using the tokenizer. Here I'm using the `AutoTokenizer` API, which will automatically load the appropriate tokenizer based on the checkpoint on the hub.

What's a bit tricky is that we also need to provide labels to the model. For multi-label text classification, this is a matrix of shape (batch_size, num_labels). Also important: this should be a tensor of floats rather than integers, otherwise PyTorch' `BCEWithLogitsLoss` (which the model will use) will complain, as explained [here](https://discuss.pytorch.org/t/multi-label-binary-classification-result-type-float-cant-be-cast-to-the-desired-output-type-long/117915/3).
"""

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()

  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)


#%%                 Midway check

example = encoded_dataset['train'][0]
print(example.keys())
# print(example)

tokenizer.decode(example['input_ids'])

print(example['labels'])

[id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]

# print("Checkpoint 2")

#%%                 Model initialization
"""Finally, we set the format of our data to PyTorch tensors. This will turn the training, validation and test sets into standard PyTorch [datasets](https://pytorch.org/docs/stable/data.html)."""

encoded_dataset.set_format("torch")

"""## Define model

Here we define a model that includes a pre-trained base (i.e. the weights from bert-base-uncased) are loaded, with a random initialized classification head (linear layer) on top. One should fine-tune this head, together with the pre-trained base on a labeled dataset.

This is also printed by the warning.

We set the `problem_type` to be "multi_label_classification", as this will make sure the appropriate loss function is used (namely [`BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)). We also make sure the output layer has `len(labels)` output neurons, and we set the id2label and label2id mappings.
"""
num_epochs = 3
batch_size = 16


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)


print("Checkpoint 3")
"""## Train the model!

We are going to train the model using HuggingFace's Trainer API. This requires us to define 2 things:

* `TrainingArguments`, which specify training hyperparameters. All options can be found in the [docs](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments). Below, we for example specify that we want to evaluate after every epoch of training, we would like to save the model every epoch, we set the learning rate, the batch size to use for training/evaluation, how many epochs to train for, and so on.
* a `Trainer` object (docs can be found [here](https://huggingface.co/transformers/main_classes/trainer.html#id1)).
"""

metric_name = "f1"



args = TrainingArguments(
    "bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

#%%                 Evaluation metrics
"""We are also going to compute metrics while training. For this, we need to define a `compute_metrics` function, that returns a dictionary with the desired metric values."""


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

print("Checkpoint 4")

"""Let's verify a batch as well as a forward pass:"""

encoded_dataset['train'][0]['labels'].type()

encoded_dataset['train']['input_ids'][0]

#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
print(outputs)


#%%                 Training
"""Let's start training!"""
model.cuda()
print("\n\nWe start training! \n\n")

start = t()
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
print(f"\n\nTraining took {t() - start} seconds, or {(t() - start)/60} minutes.\n\n")
"""## Evaluate

After training, we evaluate our model on the validation set.
"""

trainer.evaluate()

# Saving model locally. Very important!
trainer.save_model("BERT finetuned on Goemotions")


#%%


results_df = pd.DataFrame(columns=["Test Sentence", "Predicted Labels", "Probabilities"])


def tester(test, confidence):
    global results_df  # Access the global DataFrame
    
    encoding = tokenizer(test, return_tensors="pt")
    encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}
    outputs = trainer.model(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    # Get indices of sentiments whose probabilities exceed the confidence level
    predictions[np.where(probs >= confidence)] = 1
    # Convert the indices to labels.
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    # Sort predicted_labels based on probabilities and get corresponding probabilities
    sorted_probs = [probs[label2id[label]].item() for label in predicted_labels]
    sorted_labels_probs = sorted(zip(predicted_labels, sorted_probs), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_probs = zip(*sorted_labels_probs)
    
    res = {test: sorted_labels}
    print(res)
    
    # Append the results to the DataFrame
    results_df = results_df.append({
        "Test Sentence": test,
        "Predicted Labels": ', '.join(sorted_labels),
        "Probabilities": [round(prob, 3) for prob in sorted_probs]
    }, ignore_index=True)


tests = ["I'm happy I can finally train a model for multi-label classification",
         "Excuse me. What in the world is happening here?",
         "For too long has our people been oppressed by these capitalist swines!",
         "Seeing you with another... my heart aches with terrible pain.",
         "Wow you're incredibly skilled!",
         "I wonder if there is a different species out there among the stars.",
         "I must avenge my brother's death!",
         "I'm stressing out about this deadline.",
         "That movie creeps me out.",
         "I am so lucky to have you in my life.",
         "On behalf of my family I thank you for these exquisite gifts.",
         "I am the best. I have big muscles and cool hair.",
         "And then it struck me: centipede literally means hundred-feet!",
         "We did it! I am very proud of what we have achieved together.",
         "After hearing that joke, we laughed for a solid five minutes.",
         "She mourns her deceased husband.",
         "I regret every word of what I said.",
         "How could I have made such a fool out of myself?",
         "We will be fine, don't worry.",
         "Whatever happens, it will be okay.",
         "Thomas had never seen such nonsense before.",
         "Life is pointless. All this suffering has no meaning.",
         "The thunder struck so close it scared the life out of me!",
         "Did you see that jump? That takes guts!",
         "This is the most beautiful sunset I have ever seen.",
         ]


confidence = 0.10
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    for test in tests:
        tester(test, confidence)

# Save the DataFrame to CSV
results_df.to_csv("results.csv", index=False)



#%%


numbers = torch.tensor([1,3, 1, -3,0.1])
softmax = torch.nn.Softmax(dim=-1)
sigmoid = torch.nn.Sigmoid()
# probs = sigmoid(numbers.squeeze().cpu())
probs = softmax(numbers.squeeze().cpu())
print(probs)
