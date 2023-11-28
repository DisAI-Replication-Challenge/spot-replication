# Replication study of paper "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer"

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

### Newsroom dataset

```bash
```

### Other datasets

```bash
cd data
python dataset_downloader.py
```

