# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
from typing import Union, Callable
from pathlib import Path
import warnings

import numpy as np

import torch
from torch.utils.data import (
    Dataset,
    RandomSampler, 
)
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset

from esm import BatchConverter

from ._sampler import BucketBatchSampler
from .ESM2Tokenizer import EsmTokenizer
from .data_collator import DataCollatorForKmerModeling


class MLMDataModule(pl.LightningDataModule):
    def __init__(self, 
        model_name_or_path: Union[str, Path], 
        k: int,
        train_file: Union[str, Path], 
        validation_file: Union[str, Path], 
        pad_to_max_length: bool,
        max_seq_length: int,
        preprocessing_num_workers: int, 
        dataloader_num_workers: int, 
        cache_dir: Union[str, Path], 
        overwrite_cache: bool, 
        mlm_probability: float,
        train_batch_size: int, 
        val_batch_size: int, 
        persistent_workers: bool = False):
        super().__init__()
        self.model_name_or_path = model_name_or_path,
        self.k = k,
        self.train_file = train_file
        self.validation_file = validation_file
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.dataloader_num_workers = dataloader_num_workers
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.persistent_workers = persistent_workers

    def setup(self, stage):
        if type(self.model_name_or_path) is tuple:
            self.model_name_or_path = self.model_name_or_path[0]
        if type(self.k) is tuple:
            self.k = self.k[0]
            
        tokenizer = EsmTokenizer.from_pretrained(self.model_name_or_path, k=self.k)
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        datasets = load_dataset(extension, data_files=data_files, cache_dir=self.cache_dir)

        column_names = datasets["train"].column_names
        text_column_name = "sequence" if "sequence" in column_names else column_names[0]

        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if self.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [line for line in examples[text_column_name]
                                    if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=self.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=[text_column_name, 'header'],
            load_from_cache_file=not self.overwrite_cache,
        )

        data_collator = DataCollatorForKmerModeling(
            tokenizer=tokenizer, mlm_probability=self.mlm_probability)

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            persistent_workers=self.persistent_workers
        )


class FastaDataset(Dataset):
    """ Read fasta format file
    Args:
        fasta_file (str): path to fasta file
        in_memory (bool): load all sequences into memory
            Default: False
    """

    def __init__(self,
        fasta_file: Union[str, Path],
        in_memory: bool = False):
        """ initialize dataset """

        fasta_file = Path(fasta_file)
        if not fasta_file.exists():
            raise FileNotFoundError(f"File not found: {fasta_file}")

        self._cache = None
        self.__file_stream = open(fasta_file, "r")
        if in_memory:
            self._cache = []
            remain = True
            while remain:
                record = self.__read_one_seq()
                if record is None:
                    remain = False
                    break
                self._cache.append(record)
        else:
            self.__indexing()

        self._in_memory = in_memory
        self._num_examples = len(self._cache)

    def __len__(self):
        """ return the number of sequences """
        return self._num_examples

    def __getitem__(self, idx: int):
        """ return sequence at index """
        if not 0 <= idx < self._num_examples:
            raise IndexError(f"Index out of range: {idx}")

        if self._in_memory:
            return self._cache[idx]
        else:
            self.__file_stream.seek(self._cache[idx])
            return self.__read_one_seq()

    def __read_one_seq(self):
        header = self.__file_stream.readline().strip()
        if not header:
            return None

        sid = header.split(' ')[0][1:]
        seq = ""
        while True:
            prev = self.__file_stream.tell()
            line = self.__file_stream.readline().strip()
            if not line or line.startswith(">"):
                self.__file_stream.seek(prev)
                break
            seq += line

        return {
            "id": sid,
            "header": header[1:],
            "seq": seq
        }

    def __indexing(self):
        self.__file_stream.seek(0)
        self._cache = []

        prev = self.__file_stream.tell()
        while True:
            line = self.__file_stream.readline()
            if not line:
                break

            if line.startswith(">"):
                self._cache.append(prev)

            prev = self.__file_stream.tell()


class COIDataSet(Dataset):
    """ COI taxonomy dataset """

    def __init__(self,
        data_file: Union[str, Path],
        meta_file: Union[str, Path],
        config_path: Union[str, Path],
        max_seq_length: int = 1024,
        in_memory: bool = False,
        phylum_dataset: bool = False):
        """ initialize dataset """
        self._data = FastaDataset(data_file, in_memory)
        self.tokenizer = EsmTokenizer.from_pretrained(Path(config_path))
        self.max_seq_length = max_seq_length
        self.batch_converter = BatchConverter(self.tokenizer, self.max_seq_length)
        self.phylum_dataset = phylum_dataset
        
        meta = np.load(meta_file, allow_pickle=True)
        self.names = meta['names']
        self.CM = meta['DAG'][()].toarray()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        record = self._data[idx]
        seq = record['seq']

        label = record['header'].split('|')[-1]
        if self.phylum_dataset:
            idx = np.where(self.names == label)[0][0]
        else:
            try:
                species = f's__{label}' 
                idx = np.where(self.names == species)[0][0]
            except:
                genus = f'g__{label}'
                idx = np.where(self.names == genus)[0][0]

        lineage = self.CM[:, idx]

        return lineage, seq

    def collate_fn(self, batch):
        labels, _ = zip(*batch)
        _, _, batch_tokens = self.batch_converter(batch)
        return torch.FloatTensor(np.array(labels)), batch_tokens

    @classmethod
    def get_num_classes(cls):
        return len(cls.names)

    @classmethod
    def convert_id_to_label(cls, idx: int):
        if not 0 <= idx < len(cls.names):
            raise IndexError(f"Index out of range: {idx}")

        return cls.names[idx]


def get_data_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sort_key: Callable = lambda x: len(x[1])):
    sampler = RandomSampler(dataset)

    batch_sampler = BucketBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        sort_key=sort_key,
        dataset=dataset
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    return data_loader


class DeepCOIDataModule(pl.LightningDataModule):
    """ Data module for HisMe model
    Args:
        data_dir (Union[str, Path]): Path to the data directory
        prefix (str): Prefix of the data files
        batch_size (int): Batch size
    """

    def __init__(self,
        train_file: Union[str, Path],
        valid_file: Union[str, Path],
        meta_file: Union[str, Path],
        config_path: Union[str, Path],
        batch_size: int,
        num_workers: int = 1,
        max_seq_length: int = 1024,
        in_memory: bool = True,
        phylum_dataset: bool = False):
        super().__init__()
        self.train_file = train_file
        self.valid_file = valid_file
        self.meta_file = meta_file
        self.config_path = config_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.in_memory = in_memory
        self.phylum_dataset = phylum_dataset

        self.trainset = None
        self.validset = None

    def setup(self, stage: str):
        self.trainset = COIDataSet(
            data_file=self.train_file, 
            meta_file=self.meta_file, 
            config_path=self.config_path,
            max_seq_length=self.max_seq_length,
            in_memory=self.in_memory,
            phylum_dataset=self.phylum_dataset,
        )
        self.validset = COIDataSet(
            data_file=self.valid_file, 
            meta_file=self.meta_file, 
            config_path=self.config_path,
            max_seq_length=self.max_seq_length,
            in_memory=self.in_memory,
            phylum_dataset=self.phylum_dataset,
        )

        print(f"Number of train sequences: {len(self.trainset)}")
        print(f"Number of validation sequences: {len(self.validset)}")

    def train_dataloader(self):
        return get_data_loader(
            self.trainset, 
            self.batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return get_data_loader(
            self.validset, 
            self.batch_size, 
            num_workers=self.num_workers
        )
