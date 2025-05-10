#!/usr/bin/env python

import os
import sys
import itertools as it
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import esm

from DeepCOI.modeling_cls import DeepCOI
from DeepCOI.data import FastaDataset
from DeepCOI.ESM2Tokenizer import EsmTokenizer

RANKS = ['phylum', 'class', 'order', 'family', 'genus', 'species'][1:]

def build_argparser():
	parser = ArgumentParser()

	parser.add_argument("--batch_size",
		type=int,
		metavar="INT",
		default=32,
		help="batch size; default 32")
	parser.add_argument("--output",
		type=str,
		metavar="FILENAME",
		help="output file name; default stdout")
	parser.add_argument("--cpu",
		action="store_true",
		default=False,
		help="compute on CPU")
	parser.add_argument("--save_probs",
		action="store_true",
		default=False,
		help="Save all probabilities for each inputs")
	parser.add_argument("--mcm",
		action="store_true",
		default=False,
		help="apply MCM for output")
	parser.add_argument("--phylum",
		action="store_true",
		default=False,
		help="classifying phylum level")

	req_group = parser.add_argument_group("required arguments")
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

	return parser

def get_taxa_info(names):
	num_taxa = {
		rank: np.char.startswith(names, f"{rank[0]}__").sum()
		for rank in RANKS
	}
	end_pos = list(it.accumulate(num_taxa.values()))
	start_pos = [0] + end_pos[:-1]

	taxa_info = {}
	for rank, spos, epos in zip(RANKS, start_pos, end_pos):
		taxa_info[rank] = (spos, epos)

	return taxa_info

def cli_main():
	parser = build_argparser()
	args = parser.parse_args()

	model = torch.load(args.model)
	model.eval()
	model.mcm = args.mcm
	if not args.cpu:
		model.cuda()
	summary(model)

	taxa_info = get_taxa_info(model.names)

	tokenizer = EsmTokenizer.from_pretrained(args.config_path)
	batch_converter = esm.BatchConverter(tokenizer, 1024)
	dataset = esm.FastaBatchedDataset.from_file(args.seq)
	batches = dataset.get_batch_indices(args.batch_size * 1024, extra_toks_per_seq=1)
	data_loader = DataLoader(dataset, collate_fn=batch_converter, batch_sampler=batches)

	opath, oname = os.path.split(args.output)
	os.makedirs(opath, exist_ok=True)

	if not args.save_probs:
		ostream = open(args.output, 'w') if args.output else sys.stdout

	tmp = {}
	with torch.no_grad():
		for _ids, _, batch_tokens in tqdm(data_loader):
			if not args.cpu:
				batch_tokens = batch_tokens.cuda()
			logits = model(input_ids=batch_tokens) 

			for _id, probs in zip(_ids, logits):
				if args.save_probs:
					tmp[_id] = probs.cpu().detach().numpy()
					continue

				output = [_id]

				if args.phylum:
					idx = torch.argmax(probs)
					name = model.names[idx]
					score = f"{torch.max(probs).item():.4f}"

					output += ['phylum', name, score]
				else:
					for rank in RANKS:
						spos, epos = taxa_info[rank]
						sub_probs = probs[spos:epos]

						idx = torch.argmax(sub_probs) + spos
						name = model.names[idx][3:]
						score = f"{torch.max(sub_probs).item():.4f}"

						output += [rank, name, score]

				print("\t".join(output), file=ostream)

	if args.save_probs:
		np.save(args.output, tmp)

	if args.output is not None and not args.save_probs:
		ostream.close()

if __name__ == "__main__":
	cli_main()
