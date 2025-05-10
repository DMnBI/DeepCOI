target="Annelida"

export CUDA_VISIBLE_DEVICES=0

./src/deepcoi fine-tune \
	--config_path configs/DeepCOI-t6-k4/ \
	--pretrained models/DeepCOI-t6-k4.pt \
	--log_dir models/DeepCOI-${target} \
	--meta_file examples/fine-tune/${target}.meta.npz \
	--train_file examples/fine-tune/${target}.train.fasta \
	--validation_file examples/fine-tune/${target}.valid.fasta \
	--pad_to_max_length \
	--batch_size 32 \
	--cache_dir cached \
	--enable_rich_progress_bar \
	--freeze_esm \
	--disable_mcm \
	--cnn_width 7 \
	--learning_rate 1e-4 \
	--max_epochs 100 \
	--patience 5

