import torch
import pandas as pd

GENERATION_PROMPTS = "data/data_sample.csv"
MODEL_PATH = "models/bart_model"
GENERATION_OUTPUT_PATH = "generations/bart_gens.csv"

print("CUDA available?", torch.cuda.is_available())
print("Num devices:", torch.cuda.device_count())

frames = ["none", "economic", "capacity", "morality", "fairness", "legality", "policy", "crime", "security", "health", "qol", "cultural", "public", "political", "external", "other"]
df = pd.read_csv(GENERATION_PROMPTS, sep="\t")
df['frame'] = df.apply(lambda x: frames[x["label"]], axis=1)
model_path = MODEL_PATH

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
generated_df.to_csv(GENERATION_OUTPUT_PATH, sep="\t", index = False)