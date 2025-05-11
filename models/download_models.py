import sys
import argparse
import subprocess as sp

from tqdm import tqdm

FILEID_MAP = {
	"t6-320": {
		"FILEID": "1nIlli1OLAQUdsO-faK__ou-bmwQZYk8s",
		"MD5SUM": "49e909fb07642c1f491e191b48784dbd"
	},
	"t12-480": {
		"FILEID": "1yIInuQHOY-eK5mJPsmcnB-h5uDmaulPt",
		"MD5SUM": "9812249cd34c8f6adcb7807ac596813c"
	},
	"phylum": {
		"FILEID": "1OniD3g_mWNeQSsov42cjGMQJS3iGFgYg",
		"MD5SUM": "c30025beec5501b416099101b562da84"
	},
	"Annelida": {
		"FILEID": "1wQs2Z9KyL-o7UmcxA51knmES_8WCRXA1",
		"MD5SUM": "32597f132e67e16b4999e415493690e6"
	},
	"Arthropoda": {
		"FILEID": "1v3d790mOrguKumEYOAg4_4TjmBubnbzb",
		"MD5SUM": "bd6b8c5f39b35fdd0ba6fbfc34025a8a"
	},
	"Chordata": {
		"FILEID": "1REk2R4cmIZnMsqtykyU0Ej2ws57kpFvz",
		"MD5SUM": "21239ad18808ab144aeec57157e093c3"
	},
	"Cnidaria": {
		"FILEID": "1lh_oi99UIqCg6JSMSQw-ED0qTlZvZE5z",
		"MD5SUM": "01153a1a25b75dc13f161214369579b5"
	},
	"Echinodermata": {
		"FILEID": "1Rub95cNkoTSx2KfvwpA7btcpVXQ3gnMN",
		"MD5SUM": "58a6e04d1cebadf5dfb3c5999cee6f1e"
	},
	"Mollusca": {
		"FILEID": "1J37pfJSXrhmP0p1I1s52WvWJRWuyGxVj",
		"MD5SUM": "542b499422b8e26548d786f9adfc7733"
	},
	"Nematoda": {
		"FILEID": "1fnNu6wZHaQ78O2PiTQI17KzVOtiBppig",
		"MD5SUM": "33d9f4c9202452acbe44f4f3c496264d"
	},
	"Platyhelminthes": {
		"FILEID": "1DKwJidN7NeOOaCunoiScj25KFtZ1tCSh",
		"MD5SUM": "916a4c90bc374043236c50918c25a999"
	},
}

PRETRAINED = ['t6-320', 't12-480']
FINETUNED = ['phylum', 'Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']

def parse_args(argv = sys.argv[1:]):
	parser = argparse.ArgumentParser()

	# optional arguments
	parser.add_argument("-o", "--out-dir",
		dest="o_dir",
		metavar="PATH",
		default=".",
		help="output directory; default .")

	# required arguments:
	req_group = parser.add_argument_group("required argumnets")
	req_group.add_argument("-d",
		dest="db",
		choices=["all", "pre-trained", "fine-tuned"] + list(FILEID_MAP.keys()),
		nargs="+",
		required=True,
		help="a list of DB names to be downloaded")

	return parser.parse_args(argv)

def main(argv = sys.argv[1:]):
	args = parse_args(argv)

	targets = args.db
	if 'all' in targets:
		targets = PRETRAINED + FINETUNED
	elif 'pre-trained' in targets:
		targets = PRETRAINED
	elif 'fine-tuned' in targets:
		targets = FINETUNED

	_ = sp.run(['mkdir', '-pv', args.o_dir])

	for model in tqdm(targets, desc = "Download models"):
		file_id = FILEID_MAP[model]["FILEID"]
		md5sum = FILEID_MAP[model]["MD5SUM"]

		# Download from Google Drive
		cnd = ['gdown', '-O', f"{args.o_dir}/DeepCOI-{model}.pt", file_id]
		_ = sp.run(cmd)

		# Get md5sum
		cmd = ["md5sum", f"{args.o_dir}/DeepCOI-{model}.pt"]
		res = sp.run(cmd, stdout = sp.PIPE)
		if(res.stdout.decode('utf-8').split(' ')[0] != md5sum):
			print(f"Download Fail... ({model})", file = sys.stderr)


	return 0

if __name__ == "__main__":
	exit(main())