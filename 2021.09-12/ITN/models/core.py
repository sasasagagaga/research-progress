
# Common libraries
import json
import math
from pathlib import Path
from itertools import islice
import logging

# from tqdm.notebook import tqdm
from tqdm.auto import tqdm

import numpy as np

# Pytorch
import torch
from torch.nn.utils.rnn import pad_sequence

# Local imports
from ..core import models_paths


def construct_model_dir(language, model, model_version):
    model_dir = f"{models_paths[language]}/{model.name_for_saving}-v{model_version}/"
    return model_dir


def save_model(
        model, optimizer, model_version, epoch,
        language, cur_iter=None, val_loss_str=None, add="last"
):
    model_dir = f"{models_paths[language]}/{model.name_for_saving}-v{model_version}/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model_path = f"{model_dir}/epoch={epoch}"
    if cur_iter is not None:
        model_path = f"{model_path}-iter={cur_iter}"
    if val_loss_str is not None:
        model_path = f"{model_path}-vloss={val_loss_str}"
    if add is not None:
        model_path = f"{model_path}-{add}"
    model_path += ".model"

    torch.save({
        "model_version": model_version,
        "epoch": epoch,
        "iter": cur_iter,
        "val_loss": val_loss_str,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, model_path)

    model_params = {param: getattr(model, param) for param in model.params_to_save}

    model_params_path = f"{model_dir}/parameters.json"
    with open(model_params_path, "w") as fout:
        json.dump(model_params, fout, indent=2)

    logging.info(f"Model saved to {model_path}.")


# TODO: Добавить возможность загрузки *лучшей* модели, а не самой последней
def load_model(
        model_class, model_version, epoch, language,
        device=None, cur_iter=None, val_loss_str=None, add="last",
        keep_optimizer=True
):
    model_dir = f"{models_paths[language]}/{model_class.name_for_saving}-v{model_version}/"

    model_params_path = f"{model_dir}/parameters.json"
    with open(model_params_path, "r") as fin:
        model_params = json.load(fin)

    model_path = f"{model_dir}/epoch={epoch}"
    if cur_iter is not None:
        model_path = f"{model_path}-iter={cur_iter}"
    if val_loss_str is not None:
        model_path = f"{model_path}-vloss={val_loss_str}"
    if add is not None:
        model_path = f"{model_path}-{add}"
    model_path += ".model"

    states = torch.load(model_path)
    if "model" not in states or "optimizer" not in states:
        states = {
            "model": states,
            "optimizer": None
        }

    model = model_class(**model_params)
    if device is not None:
        model = model.to(device)
    model.load_state_dict(states["model"])
    model.eval()

    logging.info(f"Model loaded from {model_path} checkpoint.")

    if not keep_optimizer:
        return model

    if states["optimizer"] is not None:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(states["optimizer"])
        return model, optimizer

    return model, None


def train_and_validate_epoch(
        model_version,
        epoch,
        model, calc_loss_clb, compute_metrics_clb, translate_clb, join_results_clb, metrics_to_calc,
        language, tokenizer, device,
        dataloader_train, dataloader_val,
        optimizer,
        writer,
        write_every_iters=200, iters_for_val="uniform",
        print_every_iters=1000, save_every_iters=1000,
        compute_metrics_every_iters=2000
):
    assert compute_metrics_every_iters % write_every_iters == 0, \
        "write_every_iters should divide compute_metrics_every_iters"
    assert print_every_iters is False or print_every_iters % write_every_iters == 0, "write_every_iters should divide print_every_iters"
    assert save_every_iters is False or save_every_iters % write_every_iters == 0, "write_every_iters should divide save_every_iters"

    if isinstance(iters_for_val, str) and iters_for_val == "uniform":
        iters_for_val = math.ceil(len(dataloader_val) / (len(dataloader_train) / write_every_iters))

    if print_every_iters is False:
        print_every_iters = 10 ** 9

    if save_every_iters is False:
        save_every_iters = 10 ** 9

    def _preprocess_batch(batch, calc_tgt=False):
        if not isinstance(batch, tuple):
            # tgt нужен для перевода (функция вида ..._translate)
            if not calc_tgt:
                return batch, None
            tgt = [
                torch.tensor(
                    [tokenizer.bos_token_id] + x["written_ids"] + [tokenizer.eos_token_id]
                )
                for x in batch
            ]
            tgt = pad_sequence(tgt, padding_value=tokenizer.pad_token_id)
            return batch, tgt
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)
        return src, tgt


    i_val = 0
    dataloader_val_iter = iter(dataloader_val)
    best_val_loss = float('inf')

    model = model.to(device)
    model.train()

    sum_losses, cnt_losses = 0, 0
    for i_train, batch in enumerate(tqdm(dataloader_train), start=1):
        src, tgt = _preprocess_batch(batch)

        optimizer.zero_grad()
        loss = calc_loss_clb(src, tgt)

        loss.backward()

        optimizer.step()
        sum_losses += loss.item()
        cnt_losses += 1

        if i_train % write_every_iters == 0:
            epoch_for_writer = epoch * len(dataloader_train) + i_train

            avg_train_loss = sum_losses / cnt_losses
            writer.add_scalar('Loss/train', avg_train_loss, epoch_for_writer)
            sum_losses, cnt_losses = 0, 0

            model.eval()
            val_metrics = []
            for batch in islice(dataloader_val_iter, iters_for_val):
                src, tgt = _preprocess_batch(batch, calc_tgt=True)

                i_val += 1

                with torch.no_grad():
                    loss = calc_loss_clb(src, tgt)

                sum_losses += loss.item()
                cnt_losses += 1

                if i_train % compute_metrics_every_iters == 0:
                    metrics = compute_metrics_clb(model, tokenizer, device, translate_clb,
                                                  src, tgt, metrics_to_calc)
                    val_metrics.append(metrics)

            if cnt_losses > 0:
                avg_val_loss = sum_losses / cnt_losses
                writer.add_scalar('Loss/val', avg_val_loss, epoch_for_writer)

                if i_train % compute_metrics_every_iters == 0:
                    val_metrics = join_results_clb(val_metrics, metrics_to_calc)
                    for metric, value in val_metrics.items():
                        writer.add_scalar(f"{metric}/val", value, epoch_for_writer)

                avg_val_loss_str = f"{avg_val_loss:5.3f}"

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_model(model, optimizer, model_version, epoch + 1, language, i_train, avg_val_loss_str, add=None)
            else:
                avg_val_loss_str = "[NO]"

            sum_losses, cnt_losses = 0, 0

            if i_train % print_every_iters == 0:
                logging.info(
                    f"Epoch {epoch + 1}, iter {i_train:5}: "
                    f"train loss = {avg_train_loss:5.3f}, val loss = {avg_val_loss_str}"
                )

            if i_train % save_every_iters == 0:
                save_model(model, optimizer, model_version, epoch + 1, language)

            model.train()

    save_model(model, optimizer, model_version, epoch + 1, language)


@torch.inference_mode()
def validate_epoch(
        epoch,
        model, calc_loss_clb, compute_metrics_clb, translate_clb, join_results_clb, metrics_to_calc,
        tokenizer, device,
        dataloader_val,
        writer,
        fraction: float = 1.0,
        disable_fraction_writer_info: bool = False
):
    fraction = round(fraction, 2)
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"`fraction` should be from interval (0, 1], but given {fraction:0.2f}")
    fraction_writer_info = "" if disable_fraction_writer_info or fraction == 1.0 else f" (frac={fraction:0.2f})"

    num_iters = int(fraction * len(dataloader_val))
    num_iters = max(1, num_iters)

    model = model.to(device)
    model.eval()
    epoch_for_writer = epoch + 1

    def _preprocess_batch(batch, calc_tgt=False):
        if not isinstance(batch, tuple):
            # tgt нужен для перевода (функция вида ..._translate)
            if not calc_tgt:
                return batch, None
            tgt = [
                torch.tensor(
                    [tokenizer.bos_token_id] + x["written_ids"] + [tokenizer.eos_token_id]
                )
                for x in batch
            ]
            tgt = pad_sequence(tgt, padding_value=tokenizer.pad_token_id)
            return batch, tgt
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)
        return src, tgt

    losses, val_metrics = [], []
    for i, batch in enumerate(tqdm(dataloader_val, total=num_iters,
                                        desc=f"Validation, {fraction:0.2f} of val dataset "
                                             f"({num_iters} / {len(dataloader_val)} batches)")):
        if i >= num_iters:
            break

        src, tgt = _preprocess_batch(batch, calc_tgt=True)

        if calc_loss_clb is not None:
            loss = calc_loss_clb(src, tgt)
            losses.append(loss.item())

        metrics = compute_metrics_clb(model, tokenizer, device, translate_clb, src, tgt, metrics_to_calc)
        val_metrics.append(metrics)

    if calc_loss_clb is not None:
        avg_val_loss = np.mean(losses)
        writer.add_scalar(f"Epoch avg loss{fraction_writer_info}/val", avg_val_loss, epoch_for_writer)
        logging.info(f"Epoch {epoch_for_writer} validation avg loss{fraction_writer_info} → {avg_val_loss}")

    val_metrics = join_results_clb(val_metrics, metrics_to_calc)
    for metric, value in val_metrics.items():
        writer.add_scalar(f"Epoch avg {metric.upper()}{fraction_writer_info}/val", value, epoch_for_writer)
        logging.info(f"Epoch {epoch_for_writer} validation avg {metric.upper()}{fraction_writer_info} → {value}")
