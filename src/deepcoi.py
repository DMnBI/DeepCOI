#!/usr/bin/env python
# DeepCOI: a Large Language Model-Driven Framework for Fast and Accurate Taxonomic Assignment in Animal Metabarcoding

__author__ = "hjgwak"
__version__ = "0.0.1"

import os
import sys
import argparse
import logging

import subprocess as sp

def build_parser():
	def data_args(subparsers):
		data_group = subparsers.add_argument_group("data arguments")
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

		return data_group

	def training_args(subparsers):
		training_group = subparsers.add_argument_group("training arguments")
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
		training_group.add_argument("--enable_rich_progress_bar",
			action="store_true",
			default=False)

		return training_group

	def pretrain_args(subparsers):
		pretrain = subparsers.add_parser("pre-train",
			help="Pretraining ESM architecture")

		# arguments relating to data
		data_group = data_args(pretrain)
		data_group.add_argument("--k",
			type=int,
			metavar="INT",
			default=4,
			help="k for tokenizing sequences; default 4")

		# arguments relating to training parameters
		training_group = training_args(pretrain)
		training_group.add_argument("--mlm_probability",
			type=float,
			metavar="FLOAT",
			default=0.15,
			help="Probability for masking tokens; default 0.15")
		
		# required arguments
		req_group = pretrain.add_argument_group("required arguments")
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

	def finetune_args(subparsers):
		finetune = subparsers.add_parser("fine-tune",
			help="Fine-tuning DeepCOI model")

		# arguments relating to data
		data_group = data_args(finetune)
		data_group.add_argument("--phylum_dataset",
			action="store_true",
			default=False,
			help="fine-tuning for phylum-level classifier")
		data_group.add_argument("--log_dir",
			type=str,
			metavar="PATH",
			default='my_model',
			help="directory name for saving logs; default my_model")

		# arguments relating to training parameters
		training_group = training_args(finetune)
		training_group.add_argument("--scheduled_lr",
			action="store_true",
			default=False,
			help="enable scheduling LR")
		training_group.add_argument("--max_epochs",
			type=int,
			default=30,
			help="maximum number of epochs; default 30")
		training_group.add_argument("--patience",
			type=int,
			metavar="INT",
			default=3,
			help="The number of epochs of patience for EarchStopping; default 3")

		# arguments for model architecture
		model_group = finetune.add_argument_group("model arguments")
		model_group.add_argument('--disable_mcm',
			action="store_true",
			default=False,
			help="disable maximum constraints module")
		model_group.add_argument('--freeze_esm',
			action="store_true",
			default=False,
			help="Freeze pre-trained model and only train classification head")
		model_group.add_argument('--cnn_width',
			type=int,
			metavar="INT",
			default=7,
			help="width of Conv1d layer; default 7")

		# required arguments
		req_group = finetune.add_argument_group("required arguments")
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

	def predict_args(subparsers):
		predict = subparsers.add_parser("predict",
			help="Predicting using DeepCOI model")

		# optional arguments
		predict.add_argument("--batch_size",
			type=int,
			metavar="INT",
			default=32,
			help="batch size; default 32")
		predict.add_argument("--output",
			type=str,
			metavar="FILENAME",
			help="output file name; default stdout")
		predict.add_argument("--cpu",
			action="store_true",
			default=False,
			help="compute on CPU")
		predict.add_argument("--save_probs",
			action="store_true",
			default=False,
			help="Save all probabilities for each inputs")
		predict.add_argument("--mcm",
			action="store_true",
			default=False,
			help="apply MCM for output")
		predict.add_argument("--phylum",
			action="store_true",
			default=False,
			help="classifying phylum level")

		# required arguments
		req_group = predict.add_argument_group("required arguments")
		req_group.add_argument("--config_path",
			type=str,
			metavar="PATH",
			required=True,
			help="A path including configuration files; required")
		req_group.add_argument("--model",
			type=str,
			metavar="MODEL",
			required=True,
			help="pretrained model of classifier; required")
		req_group.add_argument("--seq",
			type=str,
			metavar="FASTA",
			required=True,
			help="input sequences to be classified; required")

	parser = argparse.ArgumentParser("DeepCOI")
	subparsers = parser.add_subparsers(dest='prog')

	pretrain_args(subparsers)
	finetune_args(subparsers)
	predict_args(subparsers)

	return parser

def run_pretrain(args, rpath):
	cmd = [f"{rpath}/DeepCOI-pretrain.py",
		# data arguments
		"--k", str(args.k),
		"--num_workers", str(args.num_workers),
		"--max_seq_length", str(args.max_seq_length),
		"--train_batch_size", str(args.batch_size),
		"--validation_batch_size", str(args.batch_size),
		"--cache_dir", args.cache_dir,
		# training arguments
		"--learning_rate", str(args.learning_rate),
		"--adam_beta1", str(args.adam_beta1),
		"--adam_beta2", str(args.adam_beta2),
		"--adam_epsilon", str(args.adam_epsilon),
		"--warmup_steps", str(args.warmup_steps),
		"--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
		"--mlm_probability", str(args.mlm_probability),
	]

	# On/Off data arguments
	cmd += ["--pad_to_max_length"] if args.pad_to_max_length else []
	cmd += ["--overwrite_cache"] if args.overwrite_cache else []

	# On/Off training arguments
	cmd += ["--enable_rich_progress_bar"] if args.enable_rich_progress_bar else []

	# required arguments
	cmd += [
		"--config_path", args.config_path,
		"--train_file", args.train_file,
		"--validation_file", args.validation_file,
	]

	_ = sp.run(cmd)

def run_finetune(args, rpath):
	cmd = [f"{rpath}/DeepCOI-finetune.py",
		# data arguments
		"--num_workers", str(args.num_workers),
		"--max_seq_length", str(args.max_seq_length),
		"--batch_size", str(args.batch_size),
		"--cache_dir", args.cache_dir,
		"--log_dir", args.log_dir,
		# training arguments
		"--learning_rate", str(args.learning_rate),
		"--adam_beta1", str(args.adam_beta1),
		"--adam_beta2", str(args.adam_beta2),
		"--adam_epsilon", str(args.adam_epsilon),
		"--warmup_steps", str(args.warmup_steps),
		"--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
		"--max_epochs", str(args.max_epochs),
		"--patience", str(args.patience),
		# model arguments
		"--cnn_width", str(args.cnn_width),
	]

	# On/Off data arguments
	cmd += ["--phylum_dataset"] if args.phylum_dataset else []
	cmd += ["--pad_to_max_length"] if args.pad_to_max_length else []
	cmd += ["--overwrite_cache"] if args.overwrite_cache else []

	# On/Off training arguments
	cmd += ["--scheduled_lr"] if args.scheduled_lr else []
	cmd += ["--enable_rich_progress_bar"] if args.enable_rich_progress_bar else []

	# On/Off model arguments
	cmd += ["--disable_mcm"] if args.disable_mcm else []
	cmd += ["--freeze_esm"] if args.freeze_esm else []

	# required arguments
	cmd += [
		"--config_path", args.config_path,
		"--pretrained", args.pretrained,
		"--meta", args.meta_file,
		"--train_file", args.train_file,
		"--validation_file", args.validation_file,
	]

	_ = sp.run(cmd)

def run_predict(args, rpath):
	cmd = [f"{rpath}/DeepCOI-predict.py",
		"--batch_size", str(args.batch_size),
	]

	cmd += ["--output", args.output] if args.output else []
	cmd += ["--cpu"] if args.cpu else []
	cmd += ["--save_probs"] if args.save_probs else []
	cmd += ["--mcm"] if args.mcm else []
	cmd += ["--phylum"] if args.phylum else []

	cmd += [
		"--config_path", args.config_path,
		"--model", args.model,
		"--seq", args.seq,
	]

	_ = sp.run(cmd)

def main(argv = sys.argv[1:]):
	parser = build_parser()
	args = parser.parse_args(argv)

	rpath, _ = os.path.split(os.path.realpath(__file__))

	if args.prog == "pre-train":
		run_pretrain(args, rpath)
	elif args.prog == "fine-tune":
		run_finetune(args, rpath)
	elif args.prog == "predict":
		run_predict(args, rpath)

	return 0

if __name__ == "__main__":
	exit(main())
	