import numpy as np
from typing import List, Optional
from transformers.data.data_collator import DataCollatorForSeq2Seq

class Paper_DataCollator(DataCollatorForSeq2Seq):
    # tokenizer: PreTrainedTokenizerBase
    # model: Optional[Any] = None
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100
    # return_tensors: str = "pt"
    def __call__(self, features, return_tensors=None):
        
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            
            for feature in features:
                # pad to labels
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature['labels']))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                    )
                else:
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features