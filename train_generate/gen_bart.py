import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_silver", help="silver corpus size suffix",
                    type=int)
#parser.add_argument("--local_rank", type=int)                    
args = parser.parse_args()
num_silver = args.num_silver

import torch
import pandas as pd

print("CUDA available?", torch.cuda.is_available())
print("Num devices:", torch.cuda.device_count())

frames = ["none", "economic", "capacity", "morality", "fairness", "legality", "policy", "crime", "security", "health", "qol", "cultural", "public", "political", "external", "other"]
df = pd.read_csv("data/test_80_20.csv", sep="\t")
df['frame'] = df.apply(lambda x: frames[x["label"]], axis=1)
model_path = "models/BART_seq2seq_80_20_" + str(num_silver)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
df["input"] = df["topic"] + " " + df["frame"]
max_input_length = 1024
max_target_length = 1024

from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

generated = []
for input in df["input"].values:
  
  input_ids = tokenizer.encode(input, truncation = True, return_tensors = "pt")
  input_ids = input_ids.to(model.device)
  sample_outputs = model.generate(  
    input_ids,
    max_length = 100,
    do_sample=True, 
    temperature=0.9,
    top_k=50, 
    top_p=0.95,
  )
  output_str = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
  print(output_str)
  generated.append(output_str) 
    
data_dict = {
    "generated": generated
}
generated_df = pd.DataFrame(data_dict)
generated_df.to_csv("generations/BART_seq2seq_80_20_" + str(num_silver) + ".csv", sep="\t", index = False)