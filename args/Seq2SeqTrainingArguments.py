from dataclasses import dataclass, field
from typing import Optional
from transformers  import Seq2SeqTrainingArguments

@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    
    seed: int = field(
        default = 42, 
        metadata={
            "help": "random seed for initialization"
        }
    )
    
    predict_with_generate: bool = field(
        default = True,
        metadata = {
            "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."
        }
    )