#!/usr/bin/env python

import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchinfo import summary

from DeepCOI.modeling_cls import DeepCOI, DeepCOITraining
from DeepCOI.data import DeepCOIDataModule


def build_argparser():
    parser = ArgumentParser()

    # data arguments
    data_group = parser.add_argument_group("data arguments")
    data_group.add_argument("--phylum_dataset",
        action="store_true",
        default=False)
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
    data_group.add_argument("--batch_size",
        type=int,
        metavar="INT",
        default=128,
        help="Batch size for training; default 128")
    data_group.add_argument("--cache_dir",
        type=str,
        metavar="PATH",
        default='.',
        help="directory for saving cache; default .")
    data_group.add_argument("--log_dir",
        type=str,
        metavar="PATH",
        default='my_model',
        help="directory name for saving logs; default my_model")

    model_group = parser.add_argument_group("model arguments")
    model_group.add_argument('--disable_mcm',
        action="store_true",
        default=False,
        help="disable maximum constraints module for albation test")
    model_group.add_argument('--freeze_esm',
        action="store_true",
        default=False,
        help="Freeze pre-trained model and only train classification head")
    model_group.add_argument('--cnn_width',
        type=int,
        metavar="INT",
        default=5,
        help="width of Conv1d layer before GRU; default 5")

    training_group = parser.add_argument_group("training arguments")
    training_group.add_argument("--scheduled_lr",
        action="store_true",
        default=False,
        help="enable scheduling LR")
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
    training_group.add_argument("--max_epochs",
        type=int,
        default=30,
        help="maximum number of epochs; default 30")
    training_group.add_argument('--warmup_steps',
        type=int,
        default=1,
        help="steps for warmup; default 1")
    training_group.add_argument("--patience",
        type=int,
        metavar="INT",
        default=3,
        help="The number of epochs of patience for EarchStopping; default 3")
    training_group.add_argument("--gradient_accumulation_steps",
        type=int,
        metavar="INT",
        default=1,
        help="The number of steps for accumulating gradients; default 1")
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
    req_group.add_argument("--pretrained",
        type=str,
        metavar="PT",
        required=True,
        help="Pre-trained ESM model to be backbone; required")
    req_group.add_argument("--meta_file",
        type=str,
        metavar="NPZ",
        required=True,
        help="meta data of taxa; required")
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
    print(f"Traget batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    bs = args.batch_size // args.gradient_accumulation_steps // n_gpus
    print(f"Actual loaded batch size for one device: {bs}")

    # ------------
    # data
    # ------------
    data_module = DeepCOIDataModule(
        train_file=args.train_file,
        valid_file=args.validation_file,
        meta_file=args.meta_file,
        config_path=args.config_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size // args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        in_memory=True,
        phylum_dataset=args.phylum_dataset,
    )

    # ------------
    # model
    # ------------
    model = DeepCOITraining(
        esm=args.pretrained,
        meta=args.meta_file,
        vocab_size=261,
        freeze_embedding=args.freeze_esm,
        cnn_width=args.cnn_width,
        lr_schedule=args.scheduled_lr,
        mcm=not args.disable_mcm,
    )
    summary(model)

    # ------------
    # training
    # ------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.0, 
        patience=args.patience, 
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        save_top_k=1, 
        filename='DeepCOI-{epoch}-{step}', 
        dirpath=f"lightning_logs/{args.log_dir}/checkpoints"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]
    if args.enable_rich_progress_bar:
        rich_progress_bar = RichProgressBar(leave=True)
        callbacks += [rich_progress_bar]
    logger = TensorBoardLogger(os.getcwd(), version=args.log_dir)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        strategy='dp' if n_gpus > 2 else None, devices=n_gpus,
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    cli_main()
