import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
import string
from collections import defaultdict

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
                .format(args.iambic_ckpt, checkpoint['epoch']))
        print('iambic model num params', num_params(topic_model))
 
    checkpoint = torch.load(args.frame_ckpt, map_location=args.device)
    model_args = checkpoint['args']
    frame_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    frame_model.load_state_dict(checkpoint['state_dict'])
    frame_model = frame_model.to(args.device)
    frame_model.eval()
    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.newline_ckpt, checkpoint['epoch']))
        print('iambic model num params', num_params(frame_model))

    while True:
        results = predict_frame(gpt_model, 
                    gpt_tokenizer, 
                    topic_model,
                    frame_model,
                    [args.input_text],
                    [args.input_topic],
                    [args.input_frame],
                    dataset_info,
                    args.precondition_topk,
                    args.topk, 
                    condition_lambda=args.condition_lambda,
                    device=args.device)
        for line in results:
            print(line)
        import pdb; pdb.set_trace()


def predict_frame(gpt_model, gpt_tokenizer, topic_model, frame_model, input_text, input_topic, input_frame, dataset_info, precondition_topk, postcondition_topk, condition_lambda=1.0, device='cuda'):
    assert len(input_text) == 1 # only do one at a time for now

    with torch.no_grad():
        batch_size = 1
        
        encoded_input = [gpt_tokenizer.encode(it, return_tensors='pt').to(device) if len(it) > 0 
                        else gpt_tokenizer.encode("<|endoftext|>", return_tensors='pt').to(device) 
                        for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)
        
        lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)

        length_cutoff = 60

        while lengths.max() < length_cutoff:
            #tokens_left = torch.LongTensor([length_cutoff - lengths.max() for _ in range(batch_size)]).to(device)

            gpt_logits = gpt_model(encoded_input)[0][:, -1, :] # batch x vocab
            top_logits, top_indices = gpt_logits.topk(precondition_topk, dim=1) # batch x topk
            #print(top_logits)
            #print(top_logits.shape)
            new_input_candidates = torch.cat([encoded_input.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1
            expanded_lengths = (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk
            #expanded_tokens_left = tokens_left.unsqueeze(1).expand(-1, precondition_topk) # batch x topk


            if condition_lambda == 0:
                topic_logits = torch.zeros_like(expanded_future_words).float()
            else:
                topic_logits = topic_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    None, # batch*topk x N
                                                    None, # N
                                                    None) # batch*topk      
                topic_logits = topic_logits[:,:,input_topic[0]]                                 
                topic_logits = topic_logits[:,-1].view(batch_size, precondition_topk) # batch x topk x N
                topic_logits = topic_logits - torch.log(1 + torch.exp(topic_logits)) # get correct log probs

            if condition_lambda == 0:
                frame_logits = torch.zeros_like(expanded_future_words).float()
            else:
                frame_logits = frame_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    None, # batch*topk x N
                                                    None, # N
                                                    None) # batch*topk      
                frame_logits = frame_logits[:,:,input_frame[0]]                                 
                frame_logits = frame_logits[:,-1].view(batch_size, precondition_topk) # batch x topk x N
                frame_logits = frame_logits - torch.log(1 + torch.exp(frame_logits)) # get correct log probs
            

            full_logits = top_logits + topic_logits * condition_lambda + frame_logits * condition_lambda # batch x topk
            post_logits, post_indices = full_logits.topk(postcondition_topk, dim=1)
            post_probs = F.softmax(post_logits, dim=1)
            index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
            next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # batch
            encoded_input = torch.cat([encoded_input, next_indices.unsqueeze(1)], dim=1) # batch x seq+1
            lengths = lengths + 1 # batch           
        return [gpt_tokenizer.decode(s[1:]) for s in encoded_input]


        for _ in range(length_cutoff): # really shouldn't have a line this long anyway
            gpt_logits = gpt_model(encoded_input)[0][:, -1, :] # batch x vocab
            gpt_logits[:, banned_tokens] = -1e8
            top_logits, top_indices = gpt_logits.topk(precondition_topk, dim=1)

            new_input_candidates = torch.cat([encoded_input.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1
            expanded_lengths = (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk
            expanded_future_words = future_words.unsqueeze(0).unsqueeze(1).expand(batch_size, precondition_topk, -1) # batch x topk x N
            candidate_syllables_to_go = []
            for candidate in new_input_candidates[0]:
                candidate_until_last_word_text = ' '.join(gpt_tokenizer.decode(candidate[previous_enc_len:]).split()[:-1])
                candidate_syllables_to_go.append(10 - count_syllables(candidate_until_last_word_text))
                # usually these are all the same, but run them all for correctness. could do more efficiently but it's not too slow anyway.
            expanded_syllables_to_go = torch.LongTensor(candidate_syllables_to_go).to(device).view(1, precondition_topk)

            if condition_lambda == 0:
                iambic_logits = torch.zeros_like(expanded_lengths).float()
            else:
                # truncate prefix because we trained on single lines
                iambic_logits = iambic_model(new_input_candidates[:, :, previous_enc_len:].flatten(0, 1), expanded_lengths.flatten(0, 1) - previous_enc_len, None, None, None)[:, -1] # batch*topk x seq+1 -> batch*topk
                iambic_logits = iambic_logits.view(batch_size, precondition_topk)
                iambic_logits = iambic_logits - torch.log(1 + torch.exp(iambic_logits))
            if condition_lambda == 0:
                rhyme_logits = torch.zeros_like(expanded_lengths).float()
            else:
                rhyme_logits = rhyme_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    expanded_future_words.flatten(0, 1), # batch*topk x N
                                                    log_probs, # N
                                                    expanded_syllables_to_go.flatten(0, 1)) # batch*topk
                rhyme_logits = rhyme_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
                rhyme_logits = rhyme_logits - torch.log(1 + torch.exp(rhyme_logits)) # batch x topk x N
                rhyme_logits = rhyme_logits.squeeze(2) # batch x topk
            if condition_lambda == 0:
                newline_logits = torch.zeros_like(expanded_lengths).float()
            else:
                newline_logits = newline_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    expanded_future_words.flatten(0, 1), # batch*topk x N
                                                    log_probs, # N
                                                    expanded_syllables_to_go.flatten(0, 1)) # batch*topk
                newline_logits = newline_logits[:, -1].view(batch_size, precondition_topk, -1) # batch x topk x N
                newline_logits = newline_logits - torch.log(1 + torch.exp(newline_logits)) # batch x topk x N
                newline_logits = newline_logits.squeeze(2) # batch x topk
            
            full_logits = top_logits + condition_lambda * iambic_logits + condition_lambda * rhyme_logits + condition_lambda * newline_logits
            post_logits, post_indices = full_logits.topk(postcondition_topk, dim=1)
            post_probs = F.softmax(post_logits, dim=1)
            index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
            next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # batch
            encoded_input = torch.cat([encoded_input, next_indices.unsqueeze(1)], dim=1) # batch x seq+1
            lengths = lengths + 1
            syllables_to_go = POETRY_LINE_SYLLABLES - count_syllables(gpt_tokenizer.decode(encoded_input[0][previous_enc_len:])) # if we get very unlucky with a partial word that the syllable counter doesn't recognize we might end early, but it's unlikely
            if syllables_to_go <= 0 and [gpt_tokenizer.decode(s) for s in encoded_input][0][-1] in PHRASE_ENDS:
                break
            if syllables_to_go < 0:
                # encoded_input = encoded_input[:, :-1]
                break

        return [gpt_tokenizer.decode(s) for s in encoded_input][0][len(current_text):]




def predict_iambic_pentameter_line(gpt_model, gpt_tokenizer, iambic_model, rhyme_model, newline_model, current_text, current_line_text, rhyme_group, dataset_info, rhyme_info, precondition_topk, postcondition_topk, banned_tokens=POETRY_BANNED_TOKENS, condition_lambda=1.0, device='cuda', length_cutoff=30):
    # TODO(poetry) delete banned tokens?
    with torch.no_grad():
        batch_size = 1

        rhyme_group_index = rhyme_info.rhyme_group2index[rhyme_group]
        future_words = torch.LongTensor([rhyme_group_index]).to(device) # 1
        log_probs = torch.Tensor([math.log(rhyme_info.rhyme_group_counts[rhyme_group] / rhyme_info.total_rhyme_groups)]).to(device) # 1

        # assumes initially all same length.
        previous_encoded_text = [gpt_tokenizer.encode(it, return_tensors='pt').to(device) for it in [current_text]]
        previous_enc_len = previous_encoded_text[0].shape[1]
        encoded_input = [gpt_tokenizer.encode(it, return_tensors='pt').to(device) for it in [current_text + current_line_text]] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)
        lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)

        line_syllable_count = count_syllables(current_line_text)
        assert line_syllable_count < POETRY_LINE_SYLLABLES # assume we started with less than one full line
        syllables_to_go = POETRY_LINE_SYLLABLES - line_syllable_count

        for _ in range(length_cutoff): # really shouldn't have a line this long anyway
            gpt_logits = gpt_model(encoded_input)[0][:, -1, :] # batch x vocab
            gpt_logits[:, banned_tokens] = -1e8
            top_logits, top_indices = gpt_logits.topk(precondition_topk, dim=1)

            new_input_candidates = torch.cat([encoded_input.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1
            expanded_lengths = (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk
            expanded_future_words = future_words.unsqueeze(0).unsqueeze(1).expand(batch_size, precondition_topk, -1) # batch x topk x N
            candidate_syllables_to_go = []
            for candidate in new_input_candidates[0]:
                candidate_until_last_word_text = ' '.join(gpt_tokenizer.decode(candidate[previous_enc_len:]).split()[:-1])
                candidate_syllables_to_go.append(10 - count_syllables(candidate_until_last_word_text))
                # usually these are all the same, but run them all for correctness. could do more efficiently but it's not too slow anyway.
            expanded_syllables_to_go = torch.LongTensor(candidate_syllables_to_go).to(device).view(1, precondition_topk)

            if condition_lambda == 0:
                iambic_logits = torch.zeros_like(expanded_lengths).float()
            else:
                # truncate prefix because we trained on single lines
                iambic_logits = iambic_model(new_input_candidates[:, :, previous_enc_len:].flatten(0, 1), expanded_lengths.flatten(0, 1) - previous_enc_len, None, None, None)[:, -1] # batch*topk x seq+1 -> batch*topk
                iambic_logits = iambic_logits.view(batch_size, precondition_topk)
                iambic_logits = iambic_logits - torch.log(1 + torch.exp(iambic_logits))
            if condition_lambda == 0:
                rhyme_logits = torch.zeros_like(expanded_lengths).float()
            else:
                rhyme_logits = rhyme_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    expanded_future_words.flatten(0, 1), # batch*topk x N
                                                    log_probs, # N
                                                    expanded_syllables_to_go.flatten(0, 1)) # batch*topk
                rhyme_logits = rhyme_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
                rhyme_logits = rhyme_logits - torch.log(1 + torch.exp(rhyme_logits)) # batch x topk x N
                rhyme_logits = rhyme_logits.squeeze(2) # batch x topk
            if condition_lambda == 0:
                newline_logits = torch.zeros_like(expanded_lengths).float()
            else:
                newline_logits = newline_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    expanded_future_words.flatten(0, 1), # batch*topk x N
                                                    log_probs, # N
                                                    expanded_syllables_to_go.flatten(0, 1)) # batch*topk
                newline_logits = newline_logits[:, -1].view(batch_size, precondition_topk, -1) # batch x topk x N
                newline_logits = newline_logits - torch.log(1 + torch.exp(newline_logits)) # batch x topk x N
                newline_logits = newline_logits.squeeze(2) # batch x topk
            
            full_logits = top_logits + condition_lambda * iambic_logits + condition_lambda * rhyme_logits + condition_lambda * newline_logits
            post_logits, post_indices = full_logits.topk(postcondition_topk, dim=1)
            post_probs = F.softmax(post_logits, dim=1)
            index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
            next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # batch
            encoded_input = torch.cat([encoded_input, next_indices.unsqueeze(1)], dim=1) # batch x seq+1
            lengths = lengths + 1
            syllables_to_go = POETRY_LINE_SYLLABLES - count_syllables(gpt_tokenizer.decode(encoded_input[0][previous_enc_len:])) # if we get very unlucky with a partial word that the syllable counter doesn't recognize we might end early, but it's unlikely
            if syllables_to_go <= 0 and [gpt_tokenizer.decode(s) for s in encoded_input][0][-1] in PHRASE_ENDS:
                break
            if syllables_to_go < 0:
                # encoded_input = encoded_input[:, :-1]
                break

        return [gpt_tokenizer.decode(s) for s in encoded_input][0][len(current_text):]


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--topic_ckpt', type=str, required=True)
    parser.add_argument('--frame_ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='gpt2-medium')

    parser.add_argument('--input_text', type=str, default=None, required=True, help='initial text')
    parser.add_argument('--input_topic', type=int, default=None, required=True, help='topic id')
    parser.add_argument('--input_frame', type=int, default=None, required=True, help='frame id')

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