# -*- coding: utf-8 -*-
import torch
import pandas as pd

TRAINING_DATA_PATH = "data/train_sample.csv"
MODEL_SAVE_PATH = "models/bart_model"

print("CUDA available?", torch.cuda.is_available())
print("Num devices:", torch.cuda.device_count())

frames = ["none", "economic", "capacity", "morality", "fairness", "legality", "policy", "crime", "security", "health", "qol", "cultural", "public", "political", "external", "other"]
df = pd.read_csv(TRAINING_DATA_PATH, sep="\t")
df['frame'] = df.apply(lambda x: frames[x["label"]], axis=1)

from datasets import Dataset
dataset = Dataset.from_pandas(df)

model_checkpoint = "facebook/bart-large"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


"""We can now call the tokenizer on all our texts. This is very simple, using the [`map`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map) method from the Datasets library. First we define a function that call the tokenizer on our texts:"""

def tokenize_function(examples):
    return tokenizer(examples["train_input"], padding='max_length', truncation=True, max_length=256)

"""Then we apply it to all the splits in our `datasets` object, using `batched=True` and 4 processes to speed up the preprocessing. We won't need the `text` column afterward, so we discard it."""

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["topic", "sentence", "label", "frame", "train_input"])

"""If we now look at an element of our datasets, we will see the text have been replaced by the `input_ids` the model will need:"""

def group_texts(examples):
    result = examples
    result["labels"] = result["input_ids"].copy()
    return result

"""First note that we duplicate the inputs for our labels. This is because the model of the 🤗 Transformers library apply the shifting to the right, so we don't need to do it manually.

Also note that by default, the `map` method will send a batch of 1,000 examples to be treated by the preprocessing function. So here, we will drop the remainder to make the concatenated tokenized texts a multiple of `block_size` every 1,000 examples. You can adjust this behavior by passing a higher batch size (which will also be processed slower). You can also speed-up the preprocessing by using multiprocessing:
"""

lm_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


"""And we can check our datasets have changed: now the samples contain chunks of `block_size` contiguous tokens, potentially spanning over several of our original texts."""

"""Now that the data has been cleaned, we're ready to instantiate our `Trainer`. We will a model:"""

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

"""And some `TrainingArguments`:"""

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    "ctrl-frame_ct",
    save_strategy = "steps",
    save_steps = 500,
    save_total_limit = 2,
    per_device_train_batch_size = 1,
    #evaluation_strategy = "epoch",
    learning_rate=1e-5,
    weight_decay=0.01,
    num_train_epochs=1,
    sharded_ddp="simple"
)

"""We pass along all of those to the `Trainer` class:"""
print("initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    #eval_dataset=lm_datasets["validation"],
)


print("model parallel?", trainer.is_model_parallel)
"""And we can train our model:"""

trainer.train()
trainer.save_model(MODEL_SAVE_PATH)
