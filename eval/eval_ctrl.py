import pandas as pd
import pickle
from rouge import Rouge 
rouge = Rouge()
from statistics import mean
import nltk
nltk.download("wordnet")
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer, score
print("start")
scorer = BERTScorer(lang="en", rescale_with_baseline=True)

GOLD_PATH = "data/input_gold_sample.csv"
GEN_PATH = "data/input_gen_sample.csv"
OUTPUT_PATH = "output/output_sample.csv"

test_data = pd.read_csv("data/input_gold_sample.csv", sep = "\t")
gen_data = pd.read_csv("data/input_gen_sample.csv", sep = "\t")

gen_data["generated"] = gen_data["generated"].str.replace('^.*?\.\.?', '', regex=True)

all_scores =  {
    "rouge-l-avg": [],
    "meteor-avg": [],
    "rouge-l-max": [],
    "meteor-max": [],
    "bert-max": []
}

        
for i in range(len(gen_data)):
    progress_string = "ctrl_ct " + str(i) + "\n"
    with open("progress.txt", "a") as file_object:
        file_object.write(progress_string)
    sentence_scores = {
        "rouge-l-avg": [],
        "meteor-avg": [],
        "rouge-l-max": 0,
        "meteor-max": 0
    }
    generated_sentence = gen_data.loc[i]["generated"]
    if len(generated_sentence) == 0:
        continue
    if generated_sentence.isupper():
        continue
    topic = test_data.loc[i]["topic"]
    label = test_data.loc[i]["label"]
    comparable = test_data[(test_data["topic"] == topic) &
              (test_data["label"] == label)]
    references = comparable["sentence"].values
    for ref in references:
        if ref.isupper():
            continue
        rouge_l_score = rouge.get_scores(generated_sentence, ref)[0]["rouge-l"]["f"]
        meteor_score_val = meteor_score(generated_sentence, ref)
        sentence_scores["rouge-l-avg"].append(rouge_l_score)
        sentence_scores["meteor-avg"].append(meteor_score_val)
        if rouge_l_score > sentence_scores["rouge-l-max"]:
            sentence_scores["rouge-l-max"] = rouge_l_score
        if meteor_score_val > sentence_scores["meteor-max"]:
            sentence_scores["meteor-max"] = meteor_score_val
    P_mul, R_mul, F_mul = scorer.score([generated_sentence], [list(references)])
    all_scores["bert-max"].append(F_mul.item())
    all_scores["rouge-l-avg"].append(mean(sentence_scores["rouge-l-avg"]))
    all_scores["meteor-avg"].append(mean(sentence_scores["meteor-avg"]))
    all_scores["rouge-l-max"].append(sentence_scores["rouge-l-max"])
    all_scores["meteor-max"].append(sentence_scores["meteor-max"])
    if (i % 100 == 0):
        temp_scores = {
            "rouge-l-avg": mean(all_scores["rouge-l-avg"]),
            "meteor-avg": mean(all_scores["meteor-avg"]),
            "rouge-l-max": mean(all_scores["rouge-l-max"]),
            "meteor-max": mean(all_scores["meteor-max"]),
            "bert-max": mean(all_scores["bert-max"])
        }            
        print("CTRL TEMP SCORES:", i)
        print(temp_scores)
    
final_scores = {
    "rouge-l-avg": mean(all_scores["rouge-l-avg"]),
    "meteor-avg": mean(all_scores["meteor-avg"]),
    "rouge-l-max": mean(all_scores["rouge-l-max"]),
    "meteor-max": mean(all_scores["meteor-max"]),
    "bert-max": mean(all_scores["bert-max"])
}


with open("progress.txt", "a") as file_object:
    file_object.write(str(final_scores))

        
print(final_scores)