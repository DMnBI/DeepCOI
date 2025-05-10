#!/usr/bin/env python

import os
import sys
import argparse
import os
import pandas as pd

from Bio import SeqIO

def parse_args(argv = sys.argv[1:]):
	arg_parser = argparse.ArgumentParser()

	# optional arguments
	arg_parser.add_argument("-o", "--output-dir",
		dest="o_dir",
		metavar="PATH",
		default=".",
		help="directory to save outputs; default .")
	arg_parser.add_argument("-c", "--cutoff",
		dest="cutoff",
		metavar="FLOAT",
		type=float,
		help="confidence score cutoff")
	arg_parser.add_argument("-t", "--targets",
		dest="targets",
		metavar="LABEL",
		nargs="*",
		help="a list of labels of target; if this option is specified, only specified labels will be reported")
	arg_parser.add_argument("-p", "--prefix",
		dest="prefix",
		metavar="STR",
		help="prefix of output files; default {prefix of query file}")

	# required arguments
	req_group = arg_parser.add_argument_group("required arguments")
	req_group.add_argument("-i", "--query",
		dest="query",
		metavar="FASTA",
		required=True,
		help="query fasta file; required")
	req_group.add_argument("-p", "--prediction",
		dest="preds",
		metavar="TSV",
		required=True,
		help="DeepCOI-phylum classification result file; required")

	return arg_parser.parse_args(argv)

def main(argv = sys.argv[1:]):
	args = parse_args(argv)

	if args.o_dir != '.':
		os.makedirs(args.o_dir, exist_ok=True)
	prefix = args.prefix or os.path.split(args.query)[-1].split('.')[0]

	preds = pd.read_csv(args.preds, sep = "\t", header=None)
	preds.columns = ['seqid', 'rank', 'label', 'score']
	records = SeqIO.index(args.query, format='fasta')

	if args.targets is not None:
		preds = preds.query("label in @args.targets")
	if args.cutoff is not None:
		preds = preds.query("score > @args.cutoff")

	for phylum in preds['label'].unique():
		ostream = open(f"{args.o_dir}/{prefix}.{phylum}.fasta", 'w')

		subdf = preds.query("label == @phylum")
		for sid in subdf['seqid']:
			print(records[sid].format('fasta'), file=ostream)

		ostream.close()

	return 0

# main
if __name__ == "__main__":
	exit(main())
