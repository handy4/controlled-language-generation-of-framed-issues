 # If there's a GPU available...
import torch

TRAIN_DATA_PATH = "data/train_sample.csv"
ADAPTER_SAVE_PATH = "model/frame_joint_final/"
ADAPTER_NAME ="frame_joint"

if torch.cuda.is_available():        # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
import pandas as pd
import sklearn

df = pd.read_csv(TRAIN_DATA_PATH, sep = "\t")
print('Number of training samples: {:,}\n'.format(df.shape[0]))
df = df.drop(columns=["topic"])
df = df.rename(columns={"sentence": "text"})

df_train = df
df_test = df.loc[len(df)*0.8:]

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

from datasets import Dataset
dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
dataset_train = dataset_train.map(encode_batch, batched=True)
dataset_test = dataset_test.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset_train.rename_column_("label", "labels")
dataset_test.rename_column_("label", "labels")
# Transform to pytorch tensors and only output the required columns
dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

from transformers import RobertaConfig, RobertaModelWithHeads

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=16,
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config
)
    
model.add_adapter(ADAPTER_NAME)
model.add_classification_head(ADAPTER_NAME, num_labels=16)
model.train_adapter(ADAPTER_NAME)

import numpy as np
from transformers import TrainingArguments, Trainer, EvalPrediction

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=14,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="model/training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_accuracy,
)

trainer.train()

MODEL_SAVE_PATH(ADAPTER_SAVE_PATH)