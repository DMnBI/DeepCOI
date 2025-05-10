export CUDA_VISIBLE_DEVICES=0

src/deepcoi predict \
	--config_path configs/DeepCOI-t6-k4/ \
	--model models/DeepCOI-Arthropoda.pt \
	--seq examples/prediction/DS-PBBC4.fasta \
	--output DS-PBBC4.tsv \
