from typing import Iterator, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from .tokenizers import ByteTokenizer


class ByteConcatDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str],
        split: str,
        text_column: str,
        seq_len: int,
        streaming: bool = True,
        shuffle_buffer_size: int = 10_000,
        seed: int = 0,
        hf_token: Optional[str] = None,
        add_doc_boundaries: bool = True,
        trust_remote_code: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.dataset_name        = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split               = split
        self.text_column         = text_column
        self.seq_len             = seq_len
        self.streaming           = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed                = seed
        self.hf_token            = hf_token
        self.add_doc_boundaries  = add_doc_boundaries
        self.trust_remote_code   = trust_remote_code
        self.rank                = rank
        self.world_size          = world_size
        self.tokenizer           = ByteTokenizer()

    def _load_hf_dataset(self):
        kwargs = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.split,
            streaming=self.streaming,
            **kwargs,
        )
        if self.streaming:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
        else:
            ds = ds.shuffle(seed=self.seed)
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        return ds

    def __iter__(self) -> Iterator[np.ndarray]:
        worker_info = torch.utils.data.get_worker_info()
        ds = self._load_hf_dataset()
        if worker_info is not None:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        buffer    = np.empty(0, dtype=np.uint8)
        block_len = self.seq_len + 1

        for example in ds:
            text = example.get(self.text_column) if isinstance(example, dict) else None
            if not text:
                continue
            encoded = self.tokenizer.encode(
                [text],
                add_bos=self.add_doc_boundaries,
                add_eos=self.add_doc_boundaries,
            )[0]["input_ids"]
            buffer = np.concatenate([buffer, encoded])
            while len(buffer) >= block_len:
                chunk  = buffer[:block_len]
                buffer = buffer[block_len:]
                yield chunk


def collate_fn(batch):
    batch     = np.stack(batch, axis=0)
    batch     = torch.from_numpy(batch.astype(np.int64))
    input_ids = batch[:, :-1].contiguous()
    targets   = batch[:, 1:].contiguous()
    return input_ids, targets
