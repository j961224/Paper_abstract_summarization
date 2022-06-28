import numpy as np
from math import ceil, floor
from typing import Union, Optional
from datasets import Dataset, DatasetDict


class DatasetTransformationNotAllowedError(Exception):
    pass

# Distribute according to the train and test ratio according to the length
class Custom_Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def train_test_split(self,
        test_size: Union[float, int, None] = None,
        train_size: Union[float, int, None] = None,
        shuffle: bool = True,
        stratify_by_column: Optional[str] = None,
        seed: Optional[int] = 42,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        train_indices_cache_file_name: Optional[str] = None,
        test_indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        train_new_fingerprint: Optional[str] = None,
        test_new_fingerprint: Optional[str] = None,
    ) -> "DatasetDict":
        from datasets import DatasetDict
        
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.train_test_split` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )
            
        # If the array is empty we do nothing
        if len(self) == 0:
            return DatasetDict({"train": self, "validation": self})
        
        if test_size is None and train_size is None:
            test_size = 0.25
        n_samples = len(self)
            
        if isinstance(test_size, float):
            n_test = ceil(test_size * n_samples)
        elif isinstance(test_size, int):
            n_test = float(test_size)

        if isinstance(train_size, float):
            n_train = floor(train_size * n_samples)
        elif isinstance(train_size, int):
            n_train = float(train_size)

        if train_size is None:
            n_train = n_samples - n_test
        elif test_size is None:
            n_test = n_samples - n_train
        n_train, n_test = int(n_train), int(n_test)
        
        # First, let's try to select 0-100 lengths, 100-200 lengths, and 200 or more lengths.
        generator = np.random.default_rng(seed) #shuffle
        
        test_0_100 = [idx for idx, entire_original_text in enumerate(self['summary_entire_orignal_text']) \
            if len(entire_original_text)>=0 and len(entire_original_text)<100]
        permutation = generator.permutation(len(test_0_100))
        n_test = int(len(test_0_100) * test_size)
        n_train = int(len(test_0_100) * (1-test_size))
        test_indices_0 = permutation[:n_test]
        train_indices_0 = permutation[n_test : (n_test + n_train)]
        
        test_100_200 = [idx for idx, entire_original_text in enumerate(self['summary_entire_orignal_text']) \
            if len(entire_original_text)>=100 and len(entire_original_text)<200]
        permutation = generator.permutation(len(test_100_200))
        n_test = int(len(test_100_200) * test_size)
        n_train = int(len(test_100_200) * (1-test_size))
        test_indices_1 = permutation[:n_test]
        train_indices_1 = permutation[n_test : (n_test + n_train)]
        
        test_over_200 = [idx for idx, entire_original_text in enumerate(self['summary_entire_orignal_text']) \
            if len(entire_original_text)>=200]
        permutation = generator.permutation(len(test_over_200))
        n_test = int(len(test_over_200) * test_size)
        n_train = int(len(test_over_200) * (1-test_size))
        test_indices_2 = permutation[:n_test]
        train_indices_2 = permutation[n_test : (n_test + n_train)]
        
        test_split = self.select(
            indices=np.append(test_indices_0, np.append(test_indices_1, test_indices_2)),
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=train_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=train_new_fingerprint,
        )
        train_split = self.select(
            indices=np.append(train_indices_0, np.append(train_indices_1, train_indices_2)),
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=test_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=test_new_fingerprint,
        )
        
        return DatasetDict({"train": train_split, "validation": test_split})
    