#!/usr/bin/env python

import sys
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from DeepCOI.modeling_mlm import EsmMLMModel
from DeepCOI.modeling_cls import DeepCOITraining

def build_argparser():
	parser = ArgumentParser()

	parser.add_argument("model",
		choices=("pre-trained", "fine-tuned"),
		help="model type to parse")
	parser.add_argument("checkpoint",
		type=str,
		help="checkpoint trained")
	parser.add_argument("output",
		type=str,
		help="output file name")

	return parser

def cli_main():
	parser = build_argparser()
	args = parser.parse_args()

	if args.model == "pre-trained":
		ckpt = args.checkpoint

		model = EsmMLMModel.load_from_checkpoint(ckpt)
		torch.save(model.model.esm, args.output)
	else:
		ckpt = args.checkpoint

		model = DeepCOITraining.load_from_checkpoint(ckpt)
		torch.save(model.model, args.output)

if __name__ == "__main__":
	cli_main()
