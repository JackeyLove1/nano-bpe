"""
"""
import torch
from cupy import mask_indices
from jaxtyping import Int
from typing import List, Tuple, Dict
from torch import Tensor

from .base import Tokenizer, get_stats, merge

def merge_torch(ids: Tensor, pair: Tensor, idx: int) -> Tensor:
    """
    GPU-accelerated version of the BPE merge function using PyTorch.

    Args:
        ids (list): The input list of token IDs.
        pair (tuple): The pair of IDs to merge.
        idx (int): The new ID to replace the pair with.

    Returns:
        list: The new list of token IDs after merging.
    """
    assert len(pair) == 2, "pair length is not 2"
    mask = (ids[:-1] == pair[0]) & (ids[1:] == pair[1])
    mask_indices = torch.where(mask)[0]

    keep_mask = torch.ones_like(ids, dtype=torch.bool)
    keep_mask[mask_indices + 1] = False

    new_ids = ids.clone()
    new_ids[mask_indices] = idx
    
    return new_ids[keep_mask]


class BasicTorchTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        dtype = torch.int16 if vocab_size < torch.iinfo(torch.int16).max else torch.int32
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"use device: {device}")
        ids_tensor = torch.tensor(ids, dtype=dtype, device=device)

        for i in range(num_merges):

            pairs: Int[torch.Tensor, "n-1 2"] = torch.stack([ids_tensor[:-1], ids_tensor[1:]], dim=1)
            uniq, counts = torch.unique(pairs, dim=0, return_counts=True)
            pair_index = torch.argmax(counts)
            pair, count = uniq[pair_index], counts[pair_index]

            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids_tensor = merge_torch(ids_tensor, pair, idx)
            # save the merge
            pair = tuple(pair.tolist())
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {count} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids