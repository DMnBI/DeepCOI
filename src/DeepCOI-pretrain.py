#!/usr/bin/env python

from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor

from DeepCOI.modeling_mlm import EsmMLMModel

from DeepCOI.data import MLMDataModule


def build_argparser():
    parser = ArgumentParser()

    # data arguments
    data_group = parser.add_argument_group("data arguments")
    data_group.add_argument("--k",
        type=int,
        metavar="INT",
        default=4,
        help="k for tokenizing sequences; default 4")
    data_group.add_argument("--pad_to_max_length",
        action="store_true",
        default=False)
    data_group.add_argument("--num_workers",
        type=int,
        metavar="INT",
        default=4,
        help="The number of workers; default 4")
    data_group.add_argument("--overwrite_cache",
        action="store_true",
        default=False)
    data_group.add_argument("--max_seq_length",
        type=int,
        metavar="INT",
        default=1024,
        help="Maximum sequence length allowed; default 1024")
    data_group.add_argument("--train_batch_size",
        type=int,
        metavar="INT",
        default=128,
        help="Batch size for training; default 128")
    data_group.add_argument("--validation_batch_size",
        type=int,
        metavar="INT",
        default=128,
        help="Batch size for validation; default 128")
    data_group.add_argument("--cache_dir",
        type=str,
        metavar="PATH",
        default='.',
        help="directory for saving cache; default .")

    training_group = parser.add_argument_group("training arguments")
    training_group.add_argument('--learning_rate', 
        type=float, 
        default=1e-4,
        help="Learning rate of AdamW; default 1e-4")
    training_group.add_argument('--adam_beta1', 
        type=float, 
        default=0.9,
        help="Adam beta1; default 0.9")
    training_group.add_argument('--adam_beta2', 
        type=float, 
        default=0.999,
        help="Adam beta2; default 0.999")
    training_group.add_argument('--adam_epsilon', 
        type=float, 
        default=1e-8,
        help="Adam epsilon; default 1e-8")
    training_group.add_argument('--warmup_steps',
        type=int,
        default=16000,
        help="steps for warmup; default 16000")
    training_group.add_argument("--gradient_accumulation_steps",
        type=int,
        metavar="INT",
        default=1,
        help="The number of steps for accumulating gradients; default 1")
    training_group.add_argument("--mlm_probability",
        type=float,
        metavar="FLOAT",
        default=0.15,
        help="Probability for masking tokens; default 0.15")
    training_group.add_argument("--enable_rich_progress_bar",
        action="store_true",
        default=False)

    # required arguments
    req_group = parser.add_argument_group("required arguments")
    req_group.add_argument("--config_path",
        type=str,
        metavar="PATH",
        required=True,
        help="A path including configuration files; required")
    req_group.add_argument("--train_file",
        type=str,
        metavar="CSV",
        required=True,
        help="Sequences for training; required")
    req_group.add_argument("--validation_file",
        type=str,
        metavar="CSV",
        required=True,
        help="Sequences for validation; required")

    return parser


def cli_main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = build_argparser()
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"No. GPUs: {n_gpus}")
    print(f"Traget batch size: {args.train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    bs = args.train_batch_size // args.gradient_accumulation_steps // n_gpus
    print(f"Actual loaded batch size for one device: {bs}")

    # ------------
    # data
    # ------------
    data_module = MLMDataModule(
        model_name_or_path=args.config_path,
        k=args.k,
        train_file=args.train_file,
        validation_file=args.validation_file,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
        train_batch_size=args.train_batch_size // args.gradient_accumulation_steps,
        val_batch_size=args.validation_batch_size // args.gradient_accumulation_steps,
        dataloader_num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        persistent_workers=True if args.num_workers > 1 else False,
    )

    # ------------
    # model
    # ------------
    lmmodel = EsmMLMModel(
        model_name_or_path=args.config_path,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )

    # ------------
    # training
    # ------------
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if args.enable_rich_progress_bar:
        rich_progress_bar = RichProgressBar(leave=True)
        callbacks += [rich_progress_bar]

    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator='gpu',
        strategy='dp' if n_gpus > 2 else None, devices=n_gpus,
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=0.1
    )
    trainer.fit(lmmodel, data_module)


if __name__ == '__main__':
    cli_main()
