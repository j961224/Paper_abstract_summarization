import os
import copy
import pprint
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import multiprocessing
from utils.metric import compute_metrics
from functools import partial
from transformers import BartTokenizerFast, Seq2SeqTrainer # , AutoConfig
from transformers import HfArgumentParser, AutoModelForSeq2SeqLM
from model.bart_model import BartForConditionalGeneration
from data.data_collator import Paper_DataCollator
from data.preprocessor import Load_data, Preprocessing, Filter, preprocess_function
from args import (CustomSeq2SeqTrainingArguments, ModelArguments, DataTrainingArguments)
    

from typing import Optional, Tuple, Dict, NoReturn, List
from tqdm import tqdm
def predict(args, model, test_dl, tokenizer) -> List[str]:
    
    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

    model.to(device)
    model.eval()
    
    pred_sentences = []
    pred_ext_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dl):
            
            gen_kwargs = args.copy()
            gen_kwargs["num_beams"] = (
                gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else model.config.num_beams
            )
                
            summary_ids = model.generate(
                input_ids=batch["input_ids"].to(device), 
                attention_mask=batch["attention_mask"].to(device),  
                **gen_kwargs,
            )
        
            # summary_sent = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            summary_sent = [tokenizer.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            pred_sentences.extend(summary_sent)

    return pred_sentences, pred_ext_ids
    # return results
