import os
import pprint
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import multiprocessing
from utils.metric import compute_metrics
from functools import partial
from transformers import BartTokenizerFast
from utils.trainer import Custom_Trainer, RL_Trainer
from transformers import HfArgumentParser, AutoModelForSeq2SeqLM
from model.bart_model import BartForConditionalGeneration
from data.data_collator import Paper_DataCollator
from data.preprocessor import Load_data, Preprocessing, Filter, preprocess_function
from args import (CustomSeq2SeqTrainingArguments, ModelArguments, DataTrainingArguments)

import nltk

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)
    
def print_elements(a: dict) -> None:
    pprint.PrettyPrinter(indent=4).pprint(a)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(training_args.seed)
    datasets = Load_data.load_train()
    
    split_datasets = datasets.train_test_split(test_size=0.2, seed=training_args.seed)
    
    
    # section data and entire data split
    train_datasets = Preprocessing.split_entire_section(dataset = split_datasets['train'])
    valid_datasets = Preprocessing.split_entire_section(dataset = split_datasets['validation'])
    
    tokenizer = BartTokenizerFast.from_pretrained(model_args.PLM)
    
    if data_args.use_preprocessing:
        preprocessor = Preprocessing()
        
        # min length down, max length over filtering
        data_filter = Filter(min_document_size=50, max_document_size=1000, min_summary_size = 10, max_summary_size = 150)
        train_datasets = train_datasets.map(preprocessor.for_train)
        train_datasets = train_datasets.filter(data_filter)
        
        valid_datasets = valid_datasets.map(preprocessor.for_train)
        valid_datasets = valid_datasets.filter(data_filter)
    
    train_size, valid_size = len(train_datasets), len(valid_datasets)
    print(train_datasets[0])
    print("-----------------------------------")
    print(valid_datasets[0])
    
    train_datasets = train_datasets.shuffle(training_args.seed)
    valid_datasets = valid_datasets.shuffle(training_args.seed)
    
    ####
    # train_datasets = train_datasets.select([0, 10, 20, 30, 40, 50])
    # valid_datasets = valid_datasets.select([0, 10, 20, 30, 40, 50])
    
    # data tokenize
    preprocess_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=data_args)
    
    train_datasets = train_datasets.map(preprocess_fn, 
        batched=True, 
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=True
    )
    
    valid_datasets = valid_datasets.map(preprocess_fn, 
        batched=True, 
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=True
    )

    # config = AutoConfig.from_pretrained(model_args.PLM)
    # config.max_length = 128
    # config.num_beams = 1
    model = BartForConditionalGeneration.from_pretrained(model_args.PLM).to(device)
    
    # -- Data Collator: dynamic padding of inputs and labels
    data_collator = Paper_DataCollator(
        tokenizer,
        model = model,
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    
    # -- compute_metrics
    compute_metric_fn  = partial(compute_metrics, tokenizer=tokenizer)
    
    # wandb connected -> performance check!
    load_dotenv(dotenv_path=log_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    if training_args.max_steps == -1:
            name = f"EP:{training_args.num_train_epochs}_"
    else:
        name = f"MS:{training_args.max_steps}_"
    name += f"RL_rougelsum_LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_{model_args.PLM}"
    
    wandb.init(
        entity="jj961224",
        project="Paper_Summarization_v0",
        name=name
    )
    wandb.config.update(training_args)
    
    print("--------------data_size-------------------")
    print(train_size, valid_size)
    print_elements(vars(training_args))
    
    training_args.predict_with_generate = True
    if training_args.do_train:
        # train code define -> not to use trainer direction
        
        use_RL = True
        # Custom_Trainer
        if use_RL == False:
            trainer = RL_Trainer(
                model,
                training_args,
                train_dataset=train_datasets,
                eval_dataset=valid_datasets,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metric_fn if training_args.predict_with_generate else None,
            )
        else:
            trainer = RL_Trainer(
                model,
                training_args,
                train_dataset=train_datasets,
                eval_dataset=valid_datasets,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metric_fn if training_args.predict_with_generate else None,
            )
        
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        model_args.save_path = os.path.join(model_args.save_path, name)
        trainer.save_model(model_args.save_path)
        ## 
        wandb.finish()

if __name__ == "__main__":
    main()