import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
from functools import partial
from utils.metric import compute
from data.data_collator import Paper_DataCollator
from data.preprocessor import Load_data, Preprocessing, Filter, preprocess_function
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from model.bart_model import BartForConditionalGeneration
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    BartTokenizerFast
)
from transformers.utils import get_full_repo_name, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_pt_utils import nested_concat
from transformers.trainer_utils import EvalPrediction


logger = get_logger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
# MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)

# summarization_name_mapping = {
#     "amazon_reviews_multi": ("review_body", "review_title"),
#     "big_patent": ("description", "abstract"),
#     "cnn_dailymail": ("article", "highlights"),
#     "orange_sum": ("text", "summary"),
#     "pn_summary": ("article", "summary"),
#     "psc": ("extract_text", "summary_text"),
#     "samsum": ("dialogue", "summary"),
#     "thaisum": ("body", "summary"),
#     "xglue": ("news_body", "news_title"),
#     "xsum": ("document", "summary"),
#     "wiki_summary": ("article", "highlights"),
# }


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=False,
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="Number of update steps between two logs if logging_strategy=step",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--PLM",
        type=str,
        default=None,
        help="Pretrained PLM name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default=None,
    #     help="Model type to use if training from scratch.",
    #     choices=MODEL_TYPES,
    # )
    # parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    # parser.add_argument(
    #     "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    # )
    # parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


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


def main():
    args = parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    args.with_tracking = True
    
    
    
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.push_to_hub:
    #         if args.hub_model_id is None:
    #             repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
    #         else:
    #             repo_name = args.hub_model_id
    #         repo = Repository(args.output_dir, clone_from=repo_name)

    #         with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
    #             if "step_*" not in gitignore:
    #                 gitignore.write("step_*\n")
    #             if "epoch_*" not in gitignore:
    #                 gitignore.write("epoch_*\n")
    #     elif args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    datasets = Load_data.load_train()
    
    split_datasets = datasets.train_test_split(test_size=0.2, seed=args.seed)
    
    
    # section data and entire data split
    train_datasets = Preprocessing.split_entire_section(dataset = split_datasets['train'])
    valid_datasets = Preprocessing.split_entire_section(dataset = split_datasets['validation'])
    
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
    print("--------------data_size-------------------")
    print(train_size, valid_size)
    
    train_datasets = train_datasets.shuffle(args.seed)
    valid_datasets = valid_datasets.shuffle(args.seed)
    
    
    tokenizer = BartTokenizerFast.from_pretrained(args.PLM)
    model = BartForConditionalGeneration.from_pretrained(args.PLM).to(device)

    # model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = train_datasets.column_names

    # Get the column names for input/target.

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False
    
    args.max_input_len = args.max_source_length
    args.max_target_len = args.max_target_length
    
    preprocess_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=args)
    
    with accelerator.main_process_first():
        train_datasets = train_datasets.map(preprocess_fn, 
            batched=True, 
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            remove_columns=column_names
        )
    
        valid_datasets = valid_datasets.map(preprocess_fn, 
            batched=True, 
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            remove_columns=column_names
        )


    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_datasets)), 1):
        logger.info(f"Sample {index} of the training set: {train_datasets[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    data_collator = Paper_DataCollator(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        # pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    

    train_dataloader = DataLoader(
        train_datasets, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(valid_datasets, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    # if hasattr(args.checkpointing_steps, "isdigit"):
    #     checkpointing_steps = args.checkpointing_steps
    #     if args.checkpointing_steps.isdigit():
    #         checkpointing_steps = int(args.checkpointing_steps)
    # else:
    #     checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("Paper_Summarization_v0", experiment_config)

    # Metric
    # compute_metric_fn  = partial(compute_metrics, tokenizer=tokenizer)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_datasets)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    metric_rouge1_for_best_model = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    model.train()
    total_loss = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        # model.train()
        if args.checkpointing_steps == "epoch" and args.with_tracking:
            total_loss = 0
            
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            # outputs = model(**batch)
            concat_inputs = {
                'input_ids': torch.cat([batch['input_ids'], batch['input_ids'].clone()], 0),
                'attention_mask': torch.cat([batch['attention_mask'], batch['attention_mask'].clone()], 0),
                'labels': torch.cat([batch['labels'], batch['labels'].clone()], 0),
            }
            
            loss = compute_loss(model, concat_inputs)
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                
            if completed_steps % args.logging_steps ==0:
                losses = total_loss.item() / args.logging_steps
                total_loss = 0
                print("==="*7+"train loss"+"==="*7)
                print("completed_steps: ",completed_steps)
                print("train_losses: ",losses)
                train_metrics = {
                    "train/loss": losses,
                    "train/epoch": completed_steps/len(train_dataloader),
                    "train/global_step": completed_steps
                }
                accelerator.log(train_metrics, step=completed_steps)

            # if isinstance(checkpointing_steps, int):
            if completed_steps % args.checkpointing_steps == 0:
                print("==="*7+"Eval start!"+"==="*7)
                output_dir = f"step_{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                
                metric_rouge1 = evaluate(model, eval_dataloader, tokenizer, accelerator, compute, completed_steps, epoch, args)
                if(metric_rouge1_for_best_model<=metric_rouge1):
                    metric_rouge1_for_best_model = metric_rouge1
                    accelerator.save_state(output_dir)
                model.train()
                

            if completed_steps >= args.max_train_steps:
                break
        
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            
            metric_rouge1 = evaluate(model, eval_dataloader, tokenizer, accelerator, compute, completed_steps, epoch, args) 
            if(metric_rouge1_for_best_model<=metric_rouge1):
                metric_rouge1_for_best_model = metric_rouge1
                accelerator.save_state(output_dir)
            model.train()
    

def evaluate(model, eval_dataloader, tokenizer, accelerator, compute, completed_steps, epoch, args):
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length,
        "num_beams": args.num_beams,
    }
    samples_seen = 0
    total_eval_loss = 0
    
    result = {"gen_len":0.0, "rouge1":0.0, "rouge2":0.0, "rougeL":0.0, "rougeLsum":0.0}
    print(len(eval_dataloader))
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            outputs = model(**batch)
            loss = outputs.loss
            if args.with_tracking:
                total_eval_loss += loss.detach().float()
            
            generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens, labels = accelerator.gather((generated_tokens, labels))
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                    decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += len(decoded_labels)

            tmp_results = compute(predictions=decoded_preds, references=decoded_labels, tokenizer=tokenizer)
            tmp_results = {key: value.mid.fmeasure * 100 for key, value in tmp_results.items()}
            
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in generated_tokens]
            tmp_results["gen_len"] = np.mean(prediction_lens)
            tmp_results = {k: round(v, 4) for k, v in result.items()}
            
            for key, value in tmp_results.items():
                ttmp = value/len(eval_dataloader)
                print(key, ttmp)
                result[key]+=ttmp
    
    # result = compute_metric_fn.compute(predictions=decoded_preds,
    #             references=decoded_labels)
    # Extract a few results from ROUGE
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    result = {k: round(v, 4) for k, v in result.items()}

    logger.info(result)

    if args.with_tracking:
        result["eval_loss"] = total_eval_loss.item() / len(eval_dataloader)
        result["epoch"] = epoch
        result["step"] = completed_steps
        accelerator.log(result, step=completed_steps)
    
    return result["rouge1"]


from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn as nn
def compute_loss(model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
    if "labels" in inputs:
        labels = inputs['labels'] # inputs.pop("labels")
        pad_mask = labels.unsqueeze(-1).eq(-100) # ignore_index
    else:
        labels = None
    
    outputs = model(**inputs)
    
    label_smoother = False
    loss = label_smoothed_nll_loss(outputs, labels,
                                        epsilon=0.1 if label_smoother else 0) # cross entropy loss로도 해보기
    
    kl_loss = compute_kl_loss(outputs, pad_mask=pad_mask)
    loss += 0.7 * kl_loss
    
    return loss

import torch.nn.functional as F
def get_normalized_probs(net_output: Dict[str, Union[torch.Tensor, Any]], log_probs=True) -> torch.Tensor:
    logits = net_output["logits"] if isinstance(net_output, dict) else net_output[0]
    if log_probs:
        return F.log_softmax(logits, dim=-1)
    else:
        return F.softmax(logits, dim=-1)


def compute_kl_loss(net_output: Dict[str, Union[torch.Tensor, Any]], pad_mask=None, reduce=True) -> torch.Tensor:
        net_prob = get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = get_normalized_probs(net_output, log_probs=False)

        p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
        p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
        
        p_loss = F.kl_div(p, q_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean')
        q_loss = F.kl_div(q, p_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean')
        
        if pad_mask is not None:
            pad_mask, _ = torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.mean()
            q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss
    
# nll loss는 softmax가 1에 가까워지면 -log(a)는 0에 가까워지므로 loss를 줄이는 것
def label_smoothed_nll_loss(model_output: Dict[str, Union[torch.Tensor, Any]], labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    log_probs = -F.log_softmax(logits, dim=-1) # ToDo 흠 이거 나눠서 log_softmax 취해야 되지 않는가
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(-100)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)
    
    nll_loss = nll_loss.sum()
    smoothed_loss = smoothed_loss.sum()
    eps_i = epsilon / log_probs.size(-1)
    return (1. - epsilon) * nll_loss + eps_i * smoothed_loss
    

if __name__ == "__main__":
    main()