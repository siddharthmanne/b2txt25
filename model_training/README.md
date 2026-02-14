# Model Training & Evaluation

This directory contains all code for training and evaluating the brain-to-text RNN phoneme decoder.

## File Descriptions

| File | Description |
|------|-------------|
| `train_model.py` | Entry point for training. Loads hyperparameters from `rnn_args.yaml`, creates a `BrainToTextDecoder_Trainer`, and runs the training loop. |
| `rnn_model.py` | Defines the `GRUDecoder` model — a multi-layer GRU with optional input patching/striding, input dropout, optional batch normalization, learnable initial hidden states, and a linear classification head that outputs phoneme logits. |
| `rnn_trainer.py` | Contains the `BrainToTextDecoder_Trainer` class which handles the full training pipeline: device setup, dataset/dataloader creation, optimizer & LR scheduler configuration, the training loop with CTC loss, periodic validation with phoneme error rate (PER) calculation, checkpointing, early stopping, and logging. |
| `dataset.py` | Defines `BrainToTextDataset`, a custom PyTorch `Dataset` that loads HDF5 neural data, organizes trials into batches (optionally sampling across multiple days), applies data augmentations, and pads/collates sequences. Also provides `train_test_split_indicies` for splitting data into train/val partitions. |
| `data_augmentations.py` | Provides the `gauss_smooth` function which applies 1D Gaussian smoothing to neural feature data along the time axis using convolution. Used during both training and inference. |
| `evaluate_model.py` | Evaluation script. Loads a pretrained `GRUDecoder` checkpoint, runs inference on specified sessions to produce phoneme logits, decodes them via greedy CTC decoding (argmax + blank/duplicate removal), computes per-trial and aggregate phoneme error rate (PER), and saves results to an `eval.txt` summary and a timestamped CSV file with predicted phoneme sequences. |
| `evaluate_model_helpers.py` | Helper utilities for evaluation: `load_h5py_file` reads HDF5 trial data (neural features, phoneme labels, metadata) into a dictionary, and `runSingleDecodingStep` smooths a neural input and runs it through the model to get logits. Also defines `LOGIT_TO_PHONEME`, the mapping from logit indices to phoneme labels. |
| `rnn_args.yaml` | YAML configuration file containing all hyperparameters for the model architecture, training schedule (learning rate, optimizer, early stopping, checkpointing), data augmentation settings, dataset paths, session lists, and evaluation settings. |
| `rnn_baseline_submission_file_valsplit.csv` | Example output CSV from a baseline model evaluation on the validation split, with columns `id` and `text` (predicted sentences). |

## Training

1. Activate the conda environment:
   ```bash
   conda activate b2txt25
   ```

2. Edit `rnn_args.yaml` to configure hyperparameters, dataset paths, and session lists as needed.

3. Run training from the `model_training` directory:
   ```bash
   python train_model.py
   ```

Training progress (loss, PER, learning rate) is logged to both stdout and a log file in the output directory. The best checkpoint is saved automatically based on validation PER. Key settings in `rnn_args.yaml`:

- `num_training_batches` — total number of training mini-batches
- `batches_per_val_step` — how often to run validation
- `output_dir` / `checkpoint_dir` — where to save model artifacts
- `early_stopping` / `early_stopping_val_steps` — stop training if no improvement

## Evaluation

1. Activate the conda environment:
   ```bash
   conda activate b2txt25
   ```

2. Configure `rnn_args.yaml` with the appropriate `model_path`, `dataset.eval_dataset` session list, and `eval_output_dir`.

3. Run evaluation:
   ```bash
   python evaluate_model.py --config_path rnn_args.yaml
   ```

The script will:
- Load the model checkpoint from `model_path`
- Run inference on all trials in the `eval_dataset` sessions
- Print per-trial and per-day phoneme error rates
- Save results to `eval_output_dir/eval.txt` and a timestamped `predicted_phonemes_*.csv`
