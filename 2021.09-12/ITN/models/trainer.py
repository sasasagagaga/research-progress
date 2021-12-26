import shutil
from argparse import Namespace
import logging

import torch

from torch.utils.tensorboard import SummaryWriter

from ..tokenizer import get_tokenizer, decode
from ..data.data_loading import get_dataloaders, CLASS_PAD_IDX
from ..metrics import join_results
from ..core import language_for_human, run_logs_dir

from .core import construct_model_dir, load_model, train_and_validate_epoch, validate_epoch


class Trainer:
    LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @staticmethod
    def _get_model_name_for_saving_and_version(model, model_version):
        if isinstance(model, dict):
            model = Namespace(**model)

        if isinstance(model, Namespace):
            if model_version is not None and "model_version" in model:
                raise ValueError("model_version specified several times")

            model_version = model.model_version
            name_for_saving = model.model_class.name_for_saving
        else:
            name_for_saving = model.name_for_saving

        return name_for_saving, model_version

    def setup_logger(self):
        file_handler = logging.FileHandler(
            f"{run_logs_dir}/{language_for_human[self.data_language]}"
            f"/{self.model_name_for_saving}-v{self.model_version}.log",
            "a"
        )
        formatter = logging.Formatter(self.LOGGING_FORMAT)
        file_handler.setFormatter(formatter)

        log = logging.getLogger()  # root logger — Good to get it only once.
        for handler in log.handlers[:]:  # remove the existing file handlers
            if isinstance(handler, logging.FileHandler):
                log.removeHandler(handler)

        log.addHandler(file_handler)  # set the new handler
        log.setLevel(logging.INFO)  # set the log level to INFO, DEBUG as the default is ERROR

    def __init__(
            self,
            model,
            data_target,

            calc_loss_iter_clb,
            compute_metrics_clb,
            translate_clb,

            batch_size,

            model_version=None,
            loss_fn=None,
            optimizer=None,
            device=torch.device("cuda:0"),
            tokenizer_name="my_ru",
            data_language="ru",
            generate_batch_clb=None,  # equals to data_target if not provided
            join_results_clb=None,    # equals to code.metrics.join_results if not provided
            metrics_to_calc="all",
            epoch=0,
            write_every_iters=200,
            iters_for_val="uniform",
            print_every_iters=False,
            save_every_iters=1000,
            compute_metrics_every_iters=2000,
            val_dataloader_fraction: float = 1.0,
            disable_val_fraction_writer_info: bool = False
    ):
        # For logging
        self.data_language = data_language
        self.model_name_for_saving, self.model_version = self._get_model_name_for_saving_and_version(model, model_version)

        self.setup_logger()

        if isinstance(model, dict):
            model = Namespace(**model)

        if isinstance(model, Namespace):
            if model_version is not None and "model_version" in model:
                raise ValueError("model_version specified several times")

            model_version = model.model_version
            if "parameters" in model:
                model = model.model_class(**model.parameters)
                model.init_weights()
            else:
                epoch = model.epoch  # model.epoch — эпоха, на которой обучение еще не проводилось (нумерация с нуля).
                # То есть после нулевой (первой по счету) эпохи обучения model.epoch = 1.
                loaded_model = load_model(model.model_class, model_version, model.epoch,
                                          data_language, device, keep_optimizer=(optimizer is None))
                if optimizer is None:
                    model, optimizer = loaded_model
                else:
                    model = loaded_model
        model = model.to(device)

        if optimizer is None:
            optimizer = model.configure_optimizer()

        if generate_batch_clb is None:
            generate_batch_clb = data_target

        if join_results_clb is None:
            join_results_clb = join_results

        tokenizer = model.tokenizer

        if loss_fn is None:
            if data_target == "seq2seq":
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            elif data_target == "classes":
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=CLASS_PAD_IDX)
            else:
                raise ValueError("loss_fn is not provided and it can be deduced from other arguments")

        local_variables = locals()
        local_variables.pop("self")
        self.__all_variable_names = list(local_variables.keys())
        for variable, value in local_variables.items():
            setattr(self, variable, value)

        dataloaders = get_dataloaders(self.tokenizer_name, self.data_language, self.data_target,
                                      self.generate_batch_clb, self.batch_size,
                                      tokenizer=self.tokenizer)
        self.dataloaders = {
            "train": dataloaders[0],
            "val": dataloaders[1]
        }

        self.writer = None
        self.refresh_writer()

    @property
    def log_dir(self):
        return f"./logs/{self.model.name_for_saving}/v{self.model_version}"

    @property
    def model_dir(self):
        return construct_model_dir(self.data_language, self.model, self.model_version)

    def refresh_writer(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def change_model_version(self, new_model_version: str):
        self.model_version = new_model_version
        self.setup_logger()
        self.refresh_writer()

    def refresh_state(self, new_model_version=None, remove_models_and_logs=False):
        if remove_models_and_logs:
            shutil.rmtree(self.log_dir, ignore_errors=True)
            shutil.rmtree(self.model_dir, ignore_errors=True)

        if new_model_version is not None:
            self.change_model_version(new_model_version)  # To update logger

        self.refresh_writer()

        self.model.init_weights()
        self.optimizer = self.model.configure_optimizer()
        self.epoch = 0

    def calc_loss(self, srcs, tgts):
        return self.calc_loss_iter_clb(srcs, tgts, self.model, self.tokenizer, self.device, self.loss_fn)

    @property
    def dataloader_train(self):
        return self.dataloaders["train"]

    @property
    def dataloader_val(self):
        return self.dataloaders["val"]

    def train_and_validate_epoch(self, write_every_iters=None, iters_for_val=None,
                                 print_every_iters=None, save_every_iters=None,
                                 compute_metrics_every_iters=None):
        if self.calc_loss_iter_clb is None:
            raise ValueError("Model can't be trained (no callback for calculating loss)")

        def new_val_if_not_none(val, new_val):
            return new_val if new_val is not None else val

        train_and_validate_epoch(
            self.model_version,
            self.epoch,
            self.model, self.calc_loss, self.compute_metrics_clb, self.translate_clb,
            self.join_results_clb, self.metrics_to_calc,
            self.data_language, self.tokenizer, self.device,
            self.dataloaders["train"], self.dataloaders["val"],
            self.optimizer,
            self.writer,
            new_val_if_not_none(self.write_every_iters, write_every_iters),
            new_val_if_not_none(self.iters_for_val, iters_for_val),
            new_val_if_not_none(self.print_every_iters, print_every_iters),
            new_val_if_not_none(self.save_every_iters, save_every_iters),
            new_val_if_not_none(self.compute_metrics_every_iters, compute_metrics_every_iters)
        )

        self.epoch += 1

    @property
    def last_train_epoch(self):
        # Номер последней эпохи, на которой обучение было закончено до конца
        return self.epoch - 1

    def validate_epoch(self, fraction=None, disable_fraction_writer_info=None):
        def new_val_if_not_none(val, new_val):
            return new_val if new_val is not None else val

        validate_epoch(
            self.last_train_epoch,
            self.model,
            None if self.calc_loss_iter_clb is None else self.calc_loss,
            self.compute_metrics_clb, self.translate_clb,
            self.join_results_clb, self.metrics_to_calc,
            self.tokenizer, self.device,
            self.dataloaders["val"],
            self.writer,
            new_val_if_not_none(self.val_dataloader_fraction, fraction),
            new_val_if_not_none(self.disable_val_fraction_writer_info, disable_fraction_writer_info)
        )

    def decode(self, sequences, batch_first=False, use_model_tokenizer=False):
        return decode(self.model.tokenizer if use_model_tokenizer else self.tokenizer,
                      sequences, batch_first=batch_first)

    def translate(self, srcs, **translate_kwargs):
        return self.translate_clb(self.model, self.tokenizer, self.device, srcs, **translate_kwargs)

    def translate_from_data(self, segment="train", return_sources=False, return_targets=True, **translate_kwargs):
        srcs, tgts = next(iter(self.dataloaders[segment]))

        result = []
        if return_sources:
            result.append(self.decode(srcs))

        result.append(self.translate(srcs, **translate_kwargs))

        if return_targets:
            result.append(self.decode(tgts))

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def calculate_metrics(self, srcs, tgts, **translate_kwargs):
        return self.compute_metrics_clb(self.model, self.tokenizer, self.device, self.translate_clb,
                                        srcs, tgts, self.metrics_to_calc, **translate_kwargs)

    def __repr__(self):
        attributes = ", ".join(
            f"{variable}={getattr(self, variable)}"
            for variable in self.__all_variable_names
            if variable != "model"
        )
        return f"{type(self.model).__name__}({attributes})"
