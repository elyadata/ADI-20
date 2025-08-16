# Arabic Dialect Identification (ADI)

## Overview
This repository contains code and scripts to train Arabic Dialect Identification (ADI) using Whisper and ECAPA-TDNN.

The project involves:
- Processing and preparing the ADI-17 and ADI-20 datasets and their subsets.
- Training and evaluating models for ADI.
- Experiment tracking and result reporting.

## Repository Structure
```
├── dataset             # Guides on how to acquire datasets
├── data_preparation    # Data preparation scripts for reprocucibility
│   └── manifests       # Datset manifests (CSV files)
└── recipes             # Modeling directory: Contains training recipes
    └── hparams         # Hyper-parameters for recipes (YAML files)
```

## Installation
### Requirements
Make sure your setup meets the following requirements.
- Python 3.10 environment
- SpeechBrain Intalled in your environment (Preferably in editable mode).
- FFMPEG or SoX (for audio processing)

To install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
The project uses the **ADI-17** and **ADI-20** datasets, which consist of Arabic dialect speech segments. Ensure that you replace the placeholder in the manifest CSVs with the location of the datasets you have downloaded.  

Pre-made manifests are available to downoad for your convenience at this [link](https://elyadata-my.sharepoint.com/:f:/p/haroun_elleuch/ErGuqCu8uXBBu0dSQu_WwmsBxwdPWoQyWfHQ67H7xav2uw?e=nk559T). All utterance segmentation is already prepared.

For licencing reasons, we do not share the audio files of the dataset. Instead, you can use the IDs of the YouTube videos to download them yourself. Only the files are needed as the segmentation and labelling of the audios are already done in the CSV manifests.  
Make sure to resample all your files to mono 16khz wav format.

## Running Experiments
### Preprocessing Audio Data
Run the preprocessing script to segment and prepare the audio files:
```bash
python data_preparation/preprocess.py --dataset_path /path/to/data
```

### Training a Model
To train a dialect classifier:
```bash
python recipes/train_<model>.py --config recipes/hparams/<experimpent_name>.yaml
```

### Evaluating a Model
To evaluate on a test set:
```bash
cd model
python train_<model_category>.py --test_only \
    --test_csv=<test_csv> \
    --eval_batch_size=8 \ 
    --fewer_eval_classes=True
```
Note that this will also generate classification reports and confusions matrices. You will find them in the `save_folder` of the model you want to evaluate. These files are also generated at the end of each epoch after validation.

## Experiment Tracking
The repository includes a script to track experiment results (validation scores only) and summarize metrics:
```bash
./results.sh            # To visulaize them in terminal
./export_results.sh     # Export results to CSV
```
This generates a table with details on best epochs, F1 scores, and precision-recall metrics.

## Citation
If using this work, please cite:
```
@inproceedings{elleuch2025adi20,
  author    = {Haroun Elleuch and Salima Mdhaffar and Yannick Estève and Fethi Bougares},
  title     = {ADI‑20: Arabic Dialect Identification Dataset and Models},
  booktitle = {Proceedings of the Annual Conference of the International Speech Communication Association (Interspeech)},
  year      = {2025},
  address   = {Rotterdam Ahoy Convention Centre, Rotterdam, The Netherlands},
  month     = {August},
  days      = {17‑21},
  note      = {To appear}
}
```