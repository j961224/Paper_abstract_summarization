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
    
    
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments)
    )
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.predict_with_generate = True
    
    datasets = Load_data.load_test()
    
    datasets = Preprocessing.split_entire_section(dataset = datasets)
    
    test_dataset = copy.deepcopy(datasets)
    if data_args.use_preprocessing:
        preprocessor = Preprocessing()
        
        data_filter = Filter(min_document_size=1, max_document_size=1000000, min_summary_size = 1, max_summary_size = 1000000)
        test_dataset = test_dataset.map(preprocessor.for_train)
        test_dataset = test_dataset.filter(data_filter)
    
    print(test_dataset)
    
    model = BartForConditionalGeneration.from_pretrained(model_args.PLM)
    tokenizer = BartTokenizerFast.from_pretrained(model_args.PLM) # PLM -> checkpoint
    
    preprocess_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=data_args)
    test_dataset = test_dataset.map(preprocess_fn, 
        batched=True, 
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=True
    )
    
    data_collator = Paper_DataCollator(
        tokenizer,
        model = model,
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    
    compute_metric_fn  = partial(compute_metrics, tokenizer=tokenizer)
    
    # trainer = Seq2SeqTrainer(
    #     model,
    #     training_args,
    #     train_dataset=None,
    #     eval_dataset=test_dataset,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metric_fn if training_args.predict_with_generate else None,
    # )
    
    # results = {}    
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    # metrics = trainer.evaluate(max_length=None, num_beams=num_beams, metric_key_prefix="eval")
    # print("#########Eval metrics: #########", metrics) 
    # metrics["eval_samples"]=len(test_dataset)
    # trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)
    
    ## predict
    
    training_args.max_length = data_args.max_target_len
    training_args.generation_max_length = data_args.max_target_len
    training_args.do_predict = True
    if training_args.do_predict:
        trainer = Seq2SeqTrainer(
            model,
            training_args,
            train_dataset=None,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metric_fn if training_args.predict_with_generate else None,
        )

        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="predict", num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    # return results
    
if __name__ == "__main__":
    main()