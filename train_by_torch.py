import os
import pprint
import wandb
import torch
import random
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
import torchmetrics
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.metric import compute_metrics
from functools import partial
from transformers import BartTokenizerFast
from transformers import HfArgumentParser, AutoModelForSeq2SeqLM
from model.bart_model import BartForConditionalGeneration
from data.data_collator import Paper_DataCollator
from data.preprocessor import Load_data, Preprocessing, Filter, preprocess_function
from args import (CustomSeq2SeqTrainingArguments, ModelArguments, DataTrainingArguments)

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

from typing import Union, List, Dict, NoReturn
from collections import defaultdict
def collate_fn(
    batched_samples: List[Dict[str, List[int]]],
    pad_token_idx: int,
    pad_keys: List[str] = ["input_ids", "labels"],
    sort_by_length: bool = True
) -> Dict[str, torch.Tensor]:
    
    if sort_by_length:
        batched_samples = sorted(batched_samples, key=lambda x: len(x["input_ids"]), reverse=True)

    keys = batched_samples[0].keys()
    outputs = defaultdict(list)

    for key in keys:
        for sample in batched_samples:
            if sample[key] is not None:
                if not isinstance(sample[key], torch.Tensor):
                    sample[key] = torch.tensor(sample[key])
                outputs[key].append(sample[key])
            else:
                outputs[key] = None
        PAD = pad_token_idx if key in pad_keys else 0
        PAD = -1 if key in "answers" else PAD
        
        if outputs[key] is not None:
            outputs[key] = torch.nn.utils.rnn.pad_sequence(outputs[key], padding_value=PAD, batch_first=True)

    return dict(outputs)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def np_sigmoid(x: np.ndarray):
    x = np.clip(x, -10, 10)
    return 1/(1+np.exp(-x))

from typing import Optional, Tuple, Dict, NoReturn
def train_step(args, model, tokenizer, batch, device) -> Tuple[torch.FloatTensor, Dict[str, float]]:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
        return_dict=True,
    )
    
    labels = batch["decoder_input_ids"][:, 1:].reshape(-1)
    logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])
    
    accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=model.config.pad_token_id)
    
    metrics = {"loss": output['loss'].item(), "logits": output['logits'].item(), 'acc':accuracy}
    
    return output['loss'].item(), metrics

def train_loop(args, model, tokenizer, train_dataloader, valid_dataloader, optimizer, scheduler, prev_step: int = 0) -> int:
    step = prev_step

    model.train()
    optimizer.zero_grad()
    losses = []
    logits = []
    accs = []
    
    import wandb
    if args.do_train:
        tqdm_bar = tqdm(train_dataloader)
        for batch in tqdm_bar:
            model.train()
            device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")
            
            loss, returned_dict = train_step(args, model, tokenizer, batch, device)
            loss.backward()
            losses.append(loss)
            logits.append(returned_dict["logits"].detach().cpu().numpy().flatten())
            accs.append(returned_dict["acc"].detach().cpu().numpy())
            step += 1
            
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                logits = np.hstack(logits)
                probs = np_sigmoid(logits)
                hist = np.histogram(probs)

                train_metrics = {
                    "train/loss": np.mean(losses),
                    "train/loss": np.mean(accs),
                    "train/probs": wandb.Histogram(np_histogram=hist),
                    "step": step,
                }
                wandb.log(train_metrics)
                losses=[]
                logits=[]
                accs=[]
            
            if args.do_eval and (step+1) % args.eval_steps == 0:
                eval(args, model, tokenizer, valid_dataloader, step)
            tqdm_bar.set_description(f"Train step {step} loss {np.mean(losses):.3f}")
        
        scheduler.step()
        
    return step

def eval(args, model, tokenizer, valid_dataloader, step) -> Dict[str, float]:
    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")
    eval_metrics = eval_loop(args, model, tokenizer, valid_dataloader, device)
    eval_metrics = {("eval/" + k): v for k, v in eval_metrics.items()}
    eval_metrics["step"] = step

    print(eval_metrics)
    import wandb
    wandb.log(eval_metrics)
    
    return eval_metrics

# def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
#     """
#     Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
#     handling potential state.
#     """
#     inputs = self._prepare_input(inputs)
#     if len(inputs) == 0:
#         raise ValueError(
#             "The batch received was empty, your model won't be able to train on it. Double-check that your "
#             f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
#         )
#     if self.args.past_index >= 0 and self._past is not None:
#         inputs["mems"] = self._past

#     return inputs

def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array

def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result

def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")

class EvalPrediction:
    # """
    # Evaluation output (always contains labels), to be used to compute metrics.
    # Parameters:
    #     predictions (`np.ndarray`): Predictions of the model.
    #     label_ids (`np.ndarray`): Targets to be matched.
    #     inputs (`np.ndarray`, *optional*)
    # """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs
        
# def nested_numpify(tensors):
#     "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
#     if isinstance(tensors, (list, tuple)):
#         return type(tensors)(nested_numpify(t) for t in tensors)
#     t = tensors.cpu()
#     if t.dtype == torch.bfloat16:
#         # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
#         # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
#         # Until Numpy adds bfloat16, we must convert float32.
#         t = t.to(torch.float32)
#     return t.numpy()


def eval_loop(args, model, tokenizer, valid_dataloader, device) -> Dict[str, float]:
    import wandb

    model.eval()
    all_losses = None
    all_preds = None
    all_labels = None
    n = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            if batch["labels"] is not None:
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    return_dict=True,
                )
                
                # inputs = _prepare_inputs(inputs)
                
                gen_kwargs = args.copy()
                gen_kwargs["num_beams"] = (
                    gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else model.config.num_beams
                )
                
                summary_ids = model.generate(
                    input_ids=batch["input_ids"].to(device), 
                    attention_mask=batch["attention_mask"].to(device),  
                    **gen_kwargs,
                )
                
                loss = (output["loss"] if isinstance(output, dict) else output[0]).mean().detach()
                
                all_losses = loss if all_losses is None else np.concatenate((all_losses, loss), axis=0)
                all_preds = summary_ids if all_preds is None else nested_concat(all_preds, summary_ids, padding_index=-100)
                all_labels = (
                        batch["labels"] if all_labels is None else nested_concat(all_labels, batch["labels"], padding_index=-100)
                    ) 
                
    metrics = compute_metrics(EvalPrediction(predictions=summary_ids, label_ids=batch["labels"]))
                
    metrics.update({"val_loss": loss.item()})
    return metrics
                
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    WANDB_AUTH_KEY = "017b4ed94106d375aeacc81e85420783fb2bea8f" 
    wandb.login(key=WANDB_AUTH_KEY)
    wandb.init(
        entity="jj961224",
        project="Paper_Summarization_v0",
        name=name
    )
    wandb.config.update(training_args)
    
    if training_args.max_steps == -1:
            name = f"EP:{training_args.num_train_epochs}_"
    else:
        name = f"MS:{training_args.max_steps}_"
    name += f"LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_{model_args.PLM}"
    
    seed_everything(training_args.seed)
    datasets = Load_data.load_train()
    
    split_datasets = datasets.train_test_split(test_size=0.2, seed=training_args.seed)
    
    # section data and entire data split
    train_datasets = Preprocessing.split_entire_section(dataset = split_datasets['train'])
    valid_datasets = Preprocessing.split_entire_section(dataset = split_datasets['validation'])
    
    tokenizer = BartTokenizerFast.from_pretrained(model_args.PLM)
    model = BartForConditionalGeneration.from_pretrained(model_args.PLM)
    
    wandb.watch(model, log='all', log_freq=500) 
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
    
    train_dataloader = DataLoader(
        train_datasets, 
        training_args.per_device_train_batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id), # 없애고도 해보기
    ) if training_args.do_train else None

    valid_dataloader = DataLoader(
        train_datasets if valid_datasets is None else valid_datasets, 
        training_args.per_device_eval_batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id),
    ) if training_args.do_eval or training_args.do_predict else None
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_args.learning_rate, 
        weight_decay=training_args.weight_decay,
        betas=[training_args.adam_beta1, training_args.adam_beta2],
    )
    total_steps = len(train_dataloader) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps, last_epoch=-1)
    
    model.train()
    model.to(device)
    optimizer.zero_grad()
    
    steps = 0
    training_args.predict_with_generate = True
    if training_args.do_train:
        for epoch in range(int(training_args.num_train_epochs)):
            print("=" * 10 + "Epoch " + str(epoch+1) + " has started! " + "=" * 10)
            steps = train_loop(training_args, model, tokenizer, train_dataloader, valid_dataloader, optimizer, scheduler, steps)
            
            model.save_pretrained(os.path.join(training_args.output_dir, f"epoch_{epoch}"))
            
            if training_args.do_predict:
                print("=" * 10 + "Epoch " + str(epoch+1) + " predict has started! " + "=" * 10)
                from inference_by_torch import predict
                pred, _ = predict(training_args, model, valid_dataloader, tokenizer)
                import json
                with open(os.path.join(training_args.output_dir, f"pred_epoch_{epoch}.json"), 'w', encoding="utf-8") as f:
                    json.dump(pred, f, ensure_ascii=False)    
            

if __name__ == "__main__":
    main()