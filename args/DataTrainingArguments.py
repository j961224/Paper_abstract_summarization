from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    use_preprocessing: bool = field(
         default=False, 
         metadata={ 
            "help": "whether to preprocess" 
        }, 
    )
    
    max_input_len: int = field(
        default = 1024,
        metadata = {
            "help":"max length of input tensor"
        }
    )
    
    max_target_len: int = field(
        default = 128,
        metadata = {
            "help":"max length of target tensor"
        }
    )
    
    pad_to_max_length: bool = field(
        default = False,
        metadata = {
            "help":(
                "Whether to apd all samples to model maximum sentence length"
            )
        }
    )
    
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )