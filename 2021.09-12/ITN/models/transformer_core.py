
# Common libraries
import math

# Pytorch
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.shape[0], :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def generate_square_subsequent_mask(size, device):
    mask = ~torch.ones((size, size), dtype=torch.bool, device=device).tril()
    return mask  # return mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, float(0.0))


def generate_padding_mask(tokenizer, src: torch.Tensor):
    return (src == tokenizer.pad_token_id).T


def create_mask(tokenizer, src: torch.Tensor = None, tgt: torch.Tensor = None):
    assert not (src is None and tgt is None), "src or tgt should be specified"

    to_return = tuple()
    if src is not None:
        src_seq_len = src.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool, device=src.device)
        src_padding_mask = generate_padding_mask(tokenizer, src)
        to_return = to_return + (src_mask, src_padding_mask)

    if tgt is not None:
        tgt_seq_len = tgt.shape[0]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        tgt_padding_mask = generate_padding_mask(tokenizer, tgt)
        to_return = to_return + (tgt_mask, tgt_padding_mask)

    return to_return
    # return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
