# Quantifying Neural Drift in Intracranial Speech Decoding

A GRU-based phoneme decoder for quantifying neural data drift in long-term intracranial speech brain-computer interfaces.

## Overview

This repository contains code for quantifying neural drift in intracranial EEG (iEEG) speech decoding over a 14-month period. We implement a five-layer GRU-based phoneme decoder trained with CTC loss to decode high-gamma neural features into phoneme sequences. Our analysis reveals systematic performance degradation over time, with phoneme error rate (PER) increasing linearly from ~58% in early sessions to ~74% in late sessions, demonstrating the critical challenge of neural drift in chronic brain-computer interfaces.

## Project Structure

- `data/` – Download the iEEG speech production dataset from [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85). More info in the README within this folder.
- `model_training/` – Model, training, dataset, and evaluation scripts for standard GRU decoder.
- `final_eval_metrics/` – Predicted phoneme CSV outputs and evaluation metrics.
- `trial_counts_summary.txt` – Summary of dataset structure across sessions.
- `requirements.txt` – Python package dependencies.
- `setup.sh` – Environment setup script.

## Methods Summary

### Task
Phoneme-level decoding from intracranial EEG (iEEG) high-gamma features extracted from 256 electrodes implanted in speech motor cortex.

### Models

- **GRU**: Five-layer GRU (hidden size = 768) with linear projection to 41 phoneme classes and CTC loss.

### Metrics
Phoneme error rate (PER) computed as `(S + D + I) / N`, where S = substitutions, D = deletions, I = insertions, and N = ground-truth phoneme count.

### Key Hyperparameters
- Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1×10⁻⁸)
- Learning rate: 1×10⁻⁴ with cosine decay to 1×10⁻⁵ over 5,000 steps
- Batch size: 64
- Dropout: 0.4 (GRU layers), 0.2 (input layer)
- Gradient clipping: max norm 10

## Dataset

This project uses the **intracranial EEG speech production dataset** from the NEJM 2024 brain-to-text study, available via [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85).

### Data Format
- **Input**: Each trial is a `T × 512` matrix of high-gamma neural features (2 features × 256 electrodes) sampled at 20ms resolution.
- **Output**: Phoneme sequence labels corresponding to spoken sentences.
- **Sessions**: 45 sessions spanning August 2023 to April 2025 (20 months).

### Splits
- **Training**: First ~15 sessions (August 2023 - October 2023): 3,436 trials
- **Validation**: From training period: 348 trials
- **Testing**: Last ~15 sessions (February 2024 - April 2025): 2,702 trials

All splits use only **Switchboard corpus** sentences to ensure consistent text distribution.

### Setup Instructions

1. **Download the dataset** from [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85):
   - Download `t15_copyTask_neuralData.zip`
   - Unzip and place the `hdf5_data_final/` folder into `data/`
   
2. **Update configuration paths** in `model_training/rnn_args.yaml` and `model_training_time_emb/rnn_args.yaml`:
   - Set `datasetPath` to point to your `data/hdf5_data_final/` directory



## Environment Setup

**Python version**: Python 3.10

**Environment manager**: conda

### Create environment

conda create -n b2txt25 python=3.10
conda activate b2txt25

text

Or using venv:
python3.10 -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows

text

### Install dependencies

pip install -r requirements.txt

text

Or install manually:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install numpy scipy scikit-learn
pip install matplotlib tqdm pyyaml h5py jiwer
pip install redis

text

The `setup.sh` script is also provided for automated setup:
./setup.sh

## How to Run
cd model_training 
Training the model: python3 train_model.py rnn_args.yaml
Evaluating the trained model: python3 evaluate_model.py --config_path rnn_args.yaml



**Underlying dataset**:
Card, N.S., Wairagkar, M., Iacobacci, C., et al. (2024).
An Accurate and Rapidly Calibrating Speech Neuroprosthesis.
New England Journal of Medicine, 391(7), 609-618.
DOI: 10.1056/NEJMoa2314132

**Dataset repository**:
Verwoert, S., et al. (2022). Intracranial EEG Speech Production Dataset.
Dryad Digital Repository. https://doi.org/10.5061/dryad.dncjsxm85


**Acknowledgments**: This project adapts code structure from the [NEJM brain-to-text repository](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text) for the Brain-to-Text '25 competition baseline.

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

