import os, random, argparse, sys, pickle, time, datasets
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import Dataset 
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

"""
This is for tutorial

If the cost is between X and Y.ipynb

These dataset creation functions are copied from
https://github.com/frankaging/align-transformers/blob/cf93a1a6491dba65e1422fe20428f5972d17137e/counterfactual_datasets/price_tagging_game.py
"""

alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""



def pricing_tag_game_config_sampler(
    amount,
    lower_bound,
    bound_width
):
    if bound_width == None:
        bound_width_sample = round(random.uniform(2.50, 7.50), 2)
    else:
        bound_width_sample = bound_width
    if lower_bound == None:
        lower_bound_sample = round(random.uniform(0.05, 9.95-bound_width_sample), 2)
        # left a little room to cover corner cases.
    else:
        lower_bound_sample = lower_bound
    upper_bound_sample = bound_width_sample + lower_bound_sample
    if amount == None:
        amount_sample = round(random.uniform(0.01, 9.99), 2)
    else:
        amount_sample = amount
        
    return lower_bound_sample, upper_bound_sample, amount_sample


def pricing_tag_game_example_sampler(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    lower_bound_sample, upper_bound_sample, amount_sample = pricing_tag_game_config_sampler(
        amount,
        lower_bound,
        bound_width
    )
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(upper_bound_str):
        # label = tokenizer.convert_tokens_to_ids("Yes")
        label = tokenizer.encode("Yes")[0]
    else:
        # label = tokenizer.convert_tokens_to_ids("No")
        label = tokenizer.encode("No")[0]

    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = alpaca_prompt_template % (instruction, amount_str)

    # nnsight change
    tokens = tokenizer.encode(alpaca_prompt)
    input_ids = torch.tensor(tokens)
    
    # input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0])*-100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    
    # assert len(input_ids) == 82
    
    return input_ids, output_ids

def factual_sampler(
    tokenizer,
    max_n_training_examples,
    game="pricing_tag",
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    
    all_input_ids = []
    all_output_ids = [] # this one does not have input ids, etc..
    for _ in range(max_n_training_examples):
        if "pricing_tag" in game:
            input_ids, output_ids = pricing_tag_game_example_sampler(
                tokenizer,
                amount,
                lower_bound,
                bound_width
            )
        elif game == "continent_retrieval":
            pass
        all_input_ids += [input_ids]
        all_output_ids += [output_ids]
        
    return all_input_ids, all_output_ids