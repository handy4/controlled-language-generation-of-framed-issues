import torch
import pandas as pd

print("CUDA available?", torch.cuda.is_available())
print("Num devices:", torch.cuda.device_count())

GENERATION_PROMPTS = "data/data_sample.csv"
MODEL_PATH = "models/ctrl_model"
GENERATION_OUTPUT_PATH = "generations/ctrl_gens.csv"

frames = ["none", "economic", "capacity", "morality", "fairness", "legality", "policy", "crime", "security", "health", "qol", "cultural", "public", "political", "external", "other"]
df = pd.read_csv(GENERATION_PROMPTS, sep="\t")
df['frame'] = df.apply(lambda x: frames[x["label"]], axis=1)
model_path = MODEL_PATH

from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
df["input"] = df["topic"] + " " + df["frame"] + "."
max_input_length = 256
max_target_length = 256

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("ctrl", use_fast=True)
tokenizer.eos_token = "Poughkeepsie,"
#tokenizer.add_special_tokens({'eos_token': '<eos>'})
tokenizer.pad_token = tokenizer.eos_token

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
    min_length = 20
  )
  output_str = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
  print(output_str)
  generated.append(output_str) 
    
data_dict = {
    "generated": generated
}
generated_df = pd.DataFrame(data_dict)
generated_df.to_csv(GENERATION_OUTPUT_PATH, sep="\t", index = False)