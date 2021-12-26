
# Pytorch
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# Local imports
from ..metrics import compute_asr_metrics, join_results
from ..tokenizer import decode, get_tokenizer

from .core import train_and_validate_epoch, validate_epoch
from .transformer_core import PositionalEncoding, TokenEmbedding, create_mask


class Seq2SeqTransformer(nn.Module):
    name_for_saving = "transformer"
    params_to_save = (
        "tokenizer_name",
        "emb_size",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout",
    )

    def __init__(
            self,
            tokenizer_name=None,
            emb_size: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 512,
            dropout: float = 0.1
    ):
        super().__init__()

        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_tokenizer(self.tokenizer_name)
        self.src_vocab_size = len(self.tokenizer)
        self.tgt_vocab_size = len(self.tokenizer)

        self.emb_size = emb_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.src_tok_emb = TokenEmbedding(self.src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(self.tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.generator = nn.Linear(emb_size, self.tgt_vocab_size)

    def forward(
            self,
            src: torch.Tensor, tgt: torch.Tensor,
            src_mask: torch.Tensor, tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor,
    ):
        memory, memory_key_padding_mask = self.encode(src, src_mask, src_padding_mask)
        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        """Returns memory and memory_key_padding_mask"""
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer_encoder(src_emb, src_mask, src_padding_mask), src_padding_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor,
               tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _resize_src_vocab_size_unsafe(self, src_vocab_size):
        """Resizes src_vocab_size in unsafe way.
        See https://huggingface.co/transformers/_modules/transformers/modeling_utils
        .html#PreTrainedModel.resize_token_embeddings for safe resizing."""
        self.src_vocab_size = src_vocab_size
        self.src_tok_emb = TokenEmbedding(self.src_vocab_size, self.emb_size)

    def _resize_src_vocab_size_to_tokenizer_size_unsafe(self):
        self._resize_src_vocab_size_unsafe(len(self.tokenizer))


def CLI_seq2seq_transformer(src, tgt, model, tokenizer, device, loss_fn):
    """CLI — Calc(ulate) Loss (for) Iter(ation)"""
    tgt_input = tgt[:-1, :]
    src_mask, src_padding_mask, tgt_mask, tgt_padding_mask = create_mask(tokenizer, src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

    tgt_output = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

    return loss


@torch.no_grad()
def seq2seq_transformer_greedy_decode(model, src, max_lengths, start_symbol, tokenizer, device):
    model = model.to(device)
    src = src.to(device)
    src_mask, src_padding_mask = create_mask(tokenizer, src)
    memory, memory_key_padding_mask = model.encode(src, src_mask, src_padding_mask)

    eos_check = torch.zeros(src.shape[1], dtype=torch.bool)
    result = torch.full((1, src.shape[1]), start_symbol, dtype=torch.long)  # always on CPU, move to GPU when needed
    for cur_len in range(1, max(max_lengths)):
        cuda_result = result.to(device)
        tgt_mask, tgt_padding_mask = create_mask(tokenizer, tgt=cuda_result)

        out = model.decode(cuda_result, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        prob = model.generator(out[-1])
        next_word = torch.argmax(prob, dim=-1).cpu()
        next_word[eos_check | (cur_len >= max_lengths)] = tokenizer.pad_token_id

        eos_check[next_word == tokenizer.eos_token_id] = 1

        result = torch.vstack([result, next_word])
        if torch.all(eos_check):
            break

    return result  # .T — for batch first


@torch.no_grad()
def seq2seq_transformer_translate(model, tokenizer, device, srcs, max_additional_tokens=5):
    model = model.to(device)
    model.eval()

    if isinstance(srcs, torch.Tensor):
        tokens = srcs
    else:
        if not isinstance(srcs, list):
            srcs = [srcs]

        batch = [torch.tensor(
            [tokenizer.bos_token_id] + tokenizer(src)["input_ids"] + [tokenizer.eos_token_id]
        ) for src in srcs]

        tokens = pad_sequence(batch, padding_value=tokenizer.pad_token_id)

    src = tokens.long()
    lengths = (src != tokenizer.pad_token_id).sum(dim=0).cpu()

    translations = seq2seq_transformer_greedy_decode(model, src, lengths + max_additional_tokens,
                                                     tokenizer.bos_token_id, tokenizer, device)

    result = decode(tokenizer, translations)
    return result


def TaVE_seq2seq_transformer(
        model_version,
        epoch,
        model,
        language, tokenizer, device,
        dataloader_train, dataloader_val,
        optimizer, loss_fn,
        writer, compute_metrics_every_iters=2000, print_every_iters=1000, write_every_iters=200, train_val_ratio=4
):
    def calc_loss_clb(src, tgt):
        return CLI_seq2seq_transformer(src, tgt, model, tokenizer, loss_fn)

    train_and_validate_epoch(
        model_version,
        epoch,
        model, calc_loss_clb, compute_asr_metrics, seq2seq_transformer_translate, join_results, "all",
        language, tokenizer, device,
        dataloader_train, dataloader_val,
        optimizer,
        writer, compute_metrics_every_iters, print_every_iters, write_every_iters, train_val_ratio
    )


@torch.no_grad()
def VE_seq2seq_transformer(epoch, model, tokenizer, device, dataloader_val, loss_fn, writer):
    def calc_loss_clb(src, tgt):
        return CLI_seq2seq_transformer(src, tgt, model, tokenizer, loss_fn)

    validate_epoch(
        epoch,
        model, calc_loss_clb, compute_asr_metrics, seq2seq_transformer_translate, join_results, "all",
        tokenizer, device,
        dataloader_val,
        writer
    )
