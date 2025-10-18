## DeepCOI: a Large Language Model-Driven Framework for Fast and Accurate Taxonomic Assignment in Animal Metabarcoding

This repository includes the implementation and experimental data of 'DeepCOI: a Large Language Model-Driven Framework for Fast and Accurate Taxonomic Assignment in Animal Metabarcoding'. Please cite our paper if you use our pipeline. Fill free to report any issue for maintenance of our model.

## Citation
If you have used DeepCOI in your research, please cite the following publication:

*In Review*


## 1. Setup
We strongly recommend you to use python virtual environment with [Anaconda](https://www.anaconda.com)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html). Moreover, this model works in practical time on GPU/TPU machine. PLEASE use one or more NVIDIA GPUs (It works without errors using just the CPU, but it will take an astronomical amount of time). The details of the machine/environment we used are as follows:

* python 3.7.16
* biopython 1.79
* cudatoolkit 11.3.1
* datasets 2.9.0
* transformers 4.26.1
* fair-esm 2.0.0
* pytorch 1.12.0
* pytorch-lightning 1.7.7
* torchinfo 1.7.2
* tqdm 4.64.1
* scipy 1.7.3


Please adjust per_device_batch_size and gradient_accumulation_steps according to the specifications of the machine you are using.

Target batch size = (No. of devices) * (per_device_batch_size) * (gradient_accumulation_steps)  
e.g. batch size 2,560 = 2 devices * 32 samples/device * 40 steps


### 1.1 Build environment
```
conda update conda (optional)
cd DeepCOI/
conda env create -f environment.yml -n deepcoi
conda activate deepcoi
```

Please make sure all the required programs are successfully installed

**NOTE.** 'cudatoolkit' and 'pytorch' packages are very sensitive to the specifications of the machine. Please make sure to install the appropriate version of 'cudatoolkit' according to the user's CUDA version, and GPU-enabled version of pytorch.


### 1.2 Download pre-trained/fine-tuned models

The source files and useful scripts are in this repository. The pre-trained and fine-tuned models have been uploaded on **Zenodo** since the size of some models is larger than 100MB. PLEASE make sure to download models after cloning this repository.

Please download the model you need through the link below and save them in the `models` directory.  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16660862.svg)](https://doi.org/10.5281/zenodo.16660862)


### 1.3 Allow executable permissions

The `DeepCOI` script in the `src` directory is an executable python script. No additional installation is required.

```
# On the top of the directory
chmod +x src/deepcoi
chmod +x src/*.py
```


## 2. How to use DeepCOI

DeepCOI consists of **THREE** main functions: **pretrain**, **finetune**, **predict**  
There is a main wrapper script `deepcoi` in the `src` directory

```
deepcoi {pretrain, finetune, predict} [options]
```
you can find details of required/optional parameters for each function with -h option.

```
deepcoi {pretrain, finetune, predict} -h
```

**\* NOTE.** DeepCOI does NOT require that data be placed in the directory where DeepCOI is installed. Just pass the location of your data as a proper parameter. However, pre-trained model should be placed in the `models/` directory.

### 2.1 Pre-train
Although a pre-trained model is given in the `models/` directory by default, you can re-train using your data.

**NOTE** Only canonical nucleotides are allowed as inputs for pre-training. Please make sure any ambiguous nucleotide is going to be included in the datasets.

**pretrain_example.sh**

```
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
```

The intermediate outputs will be stored in `lightning_logs/version_#` directory. You can extract the trained model using `src/save_trained_model.py`

### 2.2 Fine-tune
Fine-tuned models used in our publication are given in the `models/` directory by default.
You can finetune the model using your data for any specific task.

**finetune_example.sh**

```
target="Annelida"

export CUDA_VISIBLE_DEVICES=0

src/deepcoi fine-tune \
    --config_path configs/DeepCOI-t6-k4/ \
    --pretrained models/DeepCOI-t6-k4.pt \
    --log_dir DeepCOI-${target} \
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
```

The intermediate outputs will be stored in `lightning_logs/{log_dir}` directory. You can extract the trained model using `src/save_trained_model.py`

### 2.3 Predict
You can simply make classifications using the fine-tuned model. DeepCOI has two options for output file format: `.tsv` and `.npy`. By default, the `.tsv` file that includes classification results for each query will be produced. It has three columns: query id, label, score. The `.npy` file includes every estimated probability for every label. The `.npy` file is going to be produced when you pass `--save_probs` option.

**predict\_phylum\_example.sh**

```
export CUDA_VISIBLE_DEVICES=0

src/deepcoi predict \
	--config_path configs/DeepCOI-t6-k4/ \
	--model models/DeepCOI-Arthropoda.pt \
	--seq examples/predict/DS-PBBC4.fasta \
	--output DS-PBBC4.phylum.tsv \
	--phylum 
```

**predict_example.sh**

```
export CUDA_VISIBLE_DEVICES=0

src/deepcoi predict \
	--config_path configs/DeepCOI-t6-k4/ \
	--model models/DeepCOI-Arthropoda.pt \
	--seq examples/predict/DS-PBBC4.fasta \
	--output DS-PBBC4.tsv 
```

`.tsv` file contains *N* rows where *N* is the number of queries. Each row involves the rank-by-rank classification results. The classified rank, label, and score will be followed after the query sequence ID.  
The output of phylum classification model contains *4* columns, and class-to-species classification model contains *16* columns.

Example of `.tsv` file:

|seqid |rank |label |score |...|rank |label |score |
|:-----|:----------|:-----|----:|:---|:---|:---|---:|
|query0 |class |Insecta |1.00 | |species |Copicerus irroratus | 0.85 |
|query1 |class |Copepoda | 0.23 | |species |Trichosirocalus horridus | 0.01|

**save\_probs\_example.sh**

```
export CUDA_VISIBLE_DEVICES=0

src/deepcoi predict \
	--config_path configs/DeepCOI-t6-k4/ \
	--model models/DeepCOI-Arthropoda.pt \
	--seq examples/predict/DS-PBBC4.fasta \
	--output DS-PBBC4.npy \
	--save_probs
```

## 3. Utility scripts

We provide some scripts for convenience.

### 2.1 src/save\_trained\_model.py 

This script is going to extract model parameters from lightning checkpoint. Users have to pass three positional arguments: **training type**, **checkpoint path**, **output path**

**USAGE**

```
# save pre-trained model
src/save_trained_model.py pre-trained lightning_logs/version_0/checkpoints/epoch\=0-step\=78.ckpt models/my_pretrain.pt

# save fine-tuned model
src/save_trained_model.py fine-tuned lightning_logs/my_model/checkpoints/DeepCOI-epoch\=6-step\=3451.ckpt models/my_fintuned.pt
```


### 3.2 src/split_data.py

To perform hierarchical classification, input queries have to be split into separated files based on their classification results. You can simply split queries into multiple files using the provided script. 

**USAGE**

```
src/split_data.py \
    -i examples/prediction/DS-PBBC4.fasta \
    -p DS-PBBC4.phylum.csv \
    -o ./DS-PBBC/phylum/ \
    -c 0.9
```

By default, this script generates output files for all labels. You can generate output files for labels of interest.

```
src/split_data.py \
    -i examples/prediction/DS-PBBC4.fasta \
    -p DS-PBBC4.phylum.csv \
    -o ./DS-PBBC/phylum/ \
    -c 0.9 \
    -t Arthropoda
```


## 4. Run example data

This is an example of DeepCOI workflow to classify the DS-PBBC(4) dataset. Example sequenced reads are given in the `examples/predict` directory.

### 4.1 Phylum-level classification

```
export CUDA_VISIBLE_DEVICES=0

src/deepcoi predict \
	--config_path configs/DeepCOI-t6-k4/ \
	--model models/DeepCOI-Arthropoda.pt \
	--seq examples/predict/DS-PBBC4.fasta \
	--output DS-PBBC/DS-PBBC4.phylum.tsv \
	--phylum 
```

The result files `DS-PBBC4.phylum.tsv` will be generated in the `DS-PBBC` directory. Among 9,037 samples, 8,970 samples are classified as `Arthropoda` and 8,876 samples exceed confidence score cutoff 0.9.

### 4.2 Class-to-Species classification

Get records classified as `Arthropoda` with high confidence score over 0.9.

```
src/split_data.py \
    -i examples/predict/DS-PBBC4.fasta \
    -p DS-PBBC/DS-PBBC4.phylum.tsv \
    -o DS-PBBC \
    -c 0.9 \
    -t Arthropoda
```

The above command generates `DS-PBBC4.Arthropoda.fasta` file in the `DS-PBBC` directory.

```
export CUDA_VISIBLE_DEVICES=0

src/deepcoi predict \
	--config_path configs/DeepCOI-t6-k4/ \
	--model models/DeepCOI-Arthropoda.pt \
	--seq DS-PBBC/DS-PBBC4.Arthropoda.fasta \
	--output DS-PBBC/DS-PBBC4.Arthropoda.tsv
```

The result files `DS-PBBC4.Arthropoda.tsv` will be generated in the `DS-PBBC` directory. 
