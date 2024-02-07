# Replication study of paper "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer"

This is a repository from the replication study of the work [SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://aclanthology.org/2022.acl-long.346/), published in the proceedings of the ACL 2022 conference. The original implementation can be found here: [google-research/prompt-tuning](https://github.com/google-research/prompt-tuning).

This replication study is a part of [Replication Challenge](https://disai.eu/replication-challenge/) organized by [DisAI](https://disai.eu/)

## Installation

### Conda environment

```bash
conda init
conda create --name spot python 3.10.13

conda activate spot
pip install -r requirements-in.txt
```

## Data preparation

### CxC dataset

```bash
git clone https://github.com/google-research-datasets/Crisscrossed-Captions.git
cd Crisscrossed-Captions/data
mkdir sis sits sts
mv sis_test_raw.csv sis_val_raw.csv sis
mv sits_test_raw.csv sits_val_raw.csv sits
mv sts_test_raw.csv sts_val_raw.csv sts

curl https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip --output coco.zip
unzip coco.zip
```

Go to Prepare CxC.ipynb and run all blocks to get final data.

### Other datasets

```bash
cd data
python dataset_downloader.py
```

## How to run

First of all, it is necessary to create `huggingface.conf` and `wandb.conf` config files based on the example files with your own keys. These files are stores in `config/` directory. 

Firstly, activate conda environment:

```bash
conda activate spot
```

Run `script.py` with the desired settings, e.g:
```
cd src

python script.py --dataset squad --use_cpeft --wandb_project t5-squad-finetune --config_path ./configs/config-base.yaml
```


## References

Tu Vu, Brian Lester, Noah Constant, Rami Al-Rfou’, and Daniel Cer. 2022. [SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://aclanthology.org/2022.acl-long.346/). In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5039–5059, Dublin, Ireland. Association for Computational Linguistics.
