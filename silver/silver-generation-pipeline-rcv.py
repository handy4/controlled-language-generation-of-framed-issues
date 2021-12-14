import argparse
from lxml.etree import ParserError
import os
import xml.etree.ElementTree as ET

from warcio.archiveiterator import ArchiveIterator
import justext
import pandas as pd
from ahocorapy.keywordtree import KeywordTree
import nltk
nltk.download("punkt", download_dir="/ukp-storage-1/speh/silver-corpus-generation/silver-venv-adapter/nltk_data")
from nltk.tokenize import word_tokenize
from nltk.data import load
tokenizer = load("tokenizers/punkt/{0}.pickle".format("english"))

ADAPTER_SAVE_PATH = "model/frame_joint_final/"

keywords = {"abortion": ["abortion", "aborticide", "feticide", "foeticide", "embryoctony"],
           "death penalty": ["death penalty", "capital punishment", "death sentence", "death warrant", "judicial murder"], #"execution"
           "minimum wage": ["minimum wage", "living wage", "base pay", "nominal wages", "minimum income", "minimum standard of living", "subsistence level"],
           "gun control": ["gun control", "arms limitation", "arms reduction"],
           "marijuana legalization": ["marijuana legalization", "marijuana legalisation", "cannabis legalization", "legality of cannabis", "decriminalize marijuana"],
           "nuclear energy": ["nuclear energy", "nuclear power", "atomic energy", "atomic power", "fission power", "fusion power", "thermonuclear energy"],
           "same sex marriage": ["same sex marriage", "same-sex marriage", "gay marriage", "homosexual marriage", "gay wedding"],
           "smoking": ["smoking", "tobacco"],
           "immigration": ["immigration", "immigrant", "immigrants", "migrant", "migrants"],
           "universal health care": ["universal health care", "universal health coverage", "universal coverage", "universal care", "quality medical services to all citizens"],
           "coal mining": ["coal mining", "extracting coal from the ground", "extraction of coal deposits"],
           "president trump": ["president trump"]
           }
           
topics = ['abortion',
     'death penalty',
     'minimum wage',
     'gun control',
     'marijuana legalization',
     'nuclear energy',
     'same sex marriage',
     'smoking',
     'immigration',
     'universal health care',
     'coal mining',
     'president trump']
 
keyword_trees = {}
for topic in topics:
    kwtree = KeywordTree(case_insensitive=True)
    for keyword in keywords[topic]:
        kwtree.add(keyword)
    kwtree.finalize()
    keyword_trees[topic] = kwtree
   
sentences_dict = {"sentence": [],
            "topic": [],
            "score": []}

         
i = 0    
for subdir, dirs, files in os.walk("rcv1-input"):
    for file in files:
        if not ".xml" in file:
            continue
        filepath = subdir + os.sep + file
        tree = ET.parse(filepath)
        root = tree.getroot()
        for child in root:
            if child.tag == "headline":
                title = child.text
            if child.tag == "text":
                text = ET.tostring(child, encoding='unicode')
                text = text.replace("<text>", "").replace("</text>", "").replace("<p>", "").replace("</p>", "")
        if title == None:
            title = ""
        if text == None:
            text = ""
        full_text = title + text
        text = full_text
        result_dict = {}
        for topic in keyword_trees:
            results = keyword_trees[topic].search_all(text)
            result_dict[topic] = len(list(results))
        max_key = max(result_dict, key=result_dict.get)
        topic = max_key
        count = result_dict[max_key]
        if count == 0:
            continue
        tokenized = word_tokenize(text)
        tokenized = [word.lower() for word in tokenized if word.isalpha()]
        tf = count / len(tokenized)
        sentences = tokenizer.tokenize(text)
        for sent in sentences:
            sentences_dict["sentence"].append(sent)
            sentences_dict["topic"].append(topic)
            sentences_dict["score"].append(tf)     
                
df = pd.DataFrame(sentences_dict)

# If there's a GPU available...
import torch
if torch.cuda.is_available():        # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df = df.rename(columns={"sentence": "text"})

from datasets import Dataset
dataset = Dataset.from_pandas(df)

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

from transformers import RobertaConfig, RobertaModelWithHeads

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=16,
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config,
)

adapter_name = model.load_adapter(ADAPTER_SAVE_PATH)
model.set_active_adapters(adapter_name)

import numpy as np
from transformers import TrainingArguments, Trainer, EvalPrediction

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=14,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="thesis/output/training_output",
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
    #train_dataset=dataset_train,
    #eval_dataset=dataset_test,
    compute_metrics=compute_accuracy,
)

preds = trainer.predict(dataset)
softmax = torch.nn.functional.softmax(torch.tensor(preds[0]), dim=1).numpy()
frame = []
confidence = []
for row in softmax:
  max_ind = np.argmax(row)
  frame.append(max_ind)
  confidence.append(row[max_ind])
  
df["frame"] = frame
df["confidence"] = confidence
final_df = df[df["frame"]!= 0]
final_df.to_csv("pipeline_output/" + "rcv1" + ".csv", sep="\t", index = None)
