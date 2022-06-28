from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    PLM: str = field(
        default = "gogamza/kobart-base-v1",
        metadata = {
            "help": "Path to pretrained model or model identifier"
        },
    )
    
    save_path: str = field(
        default = "checkpoints",
        metadata = {
            "help": "save model path"
        },
    )