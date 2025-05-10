export CUDA_VISIBLE_DEVICES=0,1

src/deepcoi pre-train \
	--pad_to_max_length \
	--batch_size 128 \
	--gradient_accumulation_steps 4 \
	--config_path configs/DeepCOI-t6-k4 \
	--train_file examples/pre-train/euk.COI-5P.train.100K.csv \
	--validation_file examples/pre-train/euk.COI-5P.valid.10K.csv \
	--num_workers 20 \
	--k 4 \
	--learning_rate 1e-4 \
	--adam_beta1 0.9 \
	--adam_beta2 0.999 \
	--adam_epsilon 1e-8 \
	--warmup_steps 16000 \
	--enable_rich_progress_bar

