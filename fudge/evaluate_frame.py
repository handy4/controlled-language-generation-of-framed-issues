import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
import string
from collections import defaultdict
import csv

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model

from data import Dataset, load_rhyme_info
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from poetry_util import get_rhymes, count_syllables
from predict_frame import predict_frame

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_string)
    gpt_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    gpt_model = AutoModelWithLMHead.from_pretrained(args.model_string).to(args.device)
    gpt_model.eval()

    checkpoint = torch.load(args.topic_ckpt, map_location=args.device)
    model_args = checkpoint['args']
    topic_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    topic_model.load_state_dict(checkpoint['state_dict'])
    topic_model = topic_model.to(args.device)
    topic_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.topic_ckpt, checkpoint['epoch']))
        print('topic model num params', num_params(topic_model))

    
    checkpoint = torch.load(args.frame_ckpt, map_location=args.device)
    model_args = checkpoint['args']
    frame_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    frame_model.load_state_dict(checkpoint['state_dict'])
    frame_model = frame_model.to(args.device)
    frame_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.frame_ckpt, checkpoint['epoch']))
        print('frame model num params', num_params(frame_model))

    with open(args.test_file, 'rb') as rf:
                data_dict = pickle.load(rf)

    all_gens = []            
    for i in range(len(data_dict["topic"])):
        gen = predict_frame(gpt_model, 
                gpt_tokenizer, 
                topic_model, 
                frame_model,
                [""], 
                [data_dict["topic_id"][i]],
                [data_dict["frame_id"][i]],
                dataset_info, 
                args.precondition_topk,
                args.topk, 
                condition_lambda=args.condition_lambda,
                device=args.device)
        print("GENERATION ", i, "TOPIC", data_dict["topic"][i], "FRAME", data_dict["frame"][i])
        for line in gen:
            print(line)
        all_gens.append((data_dict["topic"][i], data_dict["frame"][i], gen))
        if(i % 50 == 0):
            with open(args.log_file, 'w') as wf:
                writer = csv.DictWriter(wf, fieldnames=['topic', 'frame', 'generation'])
                writer.writeheader()
                for gen_group in all_gens:
                    writer.writerow({'topic': gen_group[0], 'frame': gen_group[1], 'generation': gen_group[2]})
    with open(args.log_file, 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=['topic', 'frame', 'generation'])
        writer.writeheader()
        for gen_group in all_gens:
            writer.writerow({'topic': gen_group[0], 'frame': gen_group[1], 'generation': gen_group[2]})
        


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--topic_ckpt', type=str, required=True)
    parser.add_argument('--frame_ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='gpt2-medium')
    parser.add_argument('--log_file', type=str, required=True, help='file to write outputs to (csv format)')


    parser.add_argument('--test_file', type=str, default=None, required=True, help='file of prefix lines for couplets')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--topk', type=int, default=50, help='consider top k outputs from gpt at each step')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)