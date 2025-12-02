"""
Evaluation script for Brain-to-Text model on test/validation sets.

After training, you have:
  - checkpoint/best_checkpoint (model weights)
  - checkpoint/args.yaml (full config)

Usage:
    # Evaluate on test set using saved config
    python evaluate_model.py --checkpoint trained_models/brain2text_audio_llm/checkpoint/best_checkpoint --split test

    # Evaluate on val set
    python evaluate_model.py --checkpoint trained_models/brain2text_audio_llm/checkpoint/best_checkpoint --split val

    # Use custom config (optional)
    python evaluate_model.py --config training_args.yaml --checkpoint path/to/checkpoint --split test

Outputs (saved to checkpoint directory):
  - results_{split}_{timestamp}.csv     # Per-sample predictions and WER
  - summary_{split}_{timestamp}.json    # Aggregate metrics for paper
  - summary_{split}_{timestamp}.txt     # Human-readable summary
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import time
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import editdistance
from collections import defaultdict

from model_complete import Brain2TextModel
from dataset import BrainToTextDataset, train_test_split_indicies
from data_augmentations import apply_data_augmentations


def compute_wer(predicted_texts, target_texts):
    """Compute Word Error Rate (WER)."""
    total_edit_distance = 0
    total_words = 0

    for pred, target in zip(predicted_texts, target_texts):
        pred_words = pred.lower().strip().split()
        target_words = target.lower().strip().split()
        ed = editdistance.eval(target_words, pred_words)
        total_edit_distance += ed
        total_words += len(target_words)

    if total_words == 0:
        return 0.0, 0, 0

    wer = total_edit_distance / total_words
    return wer, total_edit_distance, total_words


def compute_cer(predicted_texts, target_texts):
    """Compute Character Error Rate (CER)."""
    total_edit_distance = 0
    total_chars = 0

    for pred, target in zip(predicted_texts, target_texts):
        pred_chars = pred.lower().strip().replace(' ', '')
        target_chars = target.lower().strip().replace(' ', '')
        ed = editdistance.eval(target_chars, pred_chars)
        total_edit_distance += ed
        total_chars += len(target_chars)

    if total_chars == 0:
        return 0.0, 0, 0

    cer = total_edit_distance / total_chars
    return cer, total_edit_distance, total_chars


def load_model(args, checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    model = Brain2TextModel(
        neural_dim=args['model']['n_input_features'],
        n_units=args['model']['n_units'],
        n_days=len(args['dataset']['sessions']),
        audio_embedding_dim=args['model']['audio_embedding_dim'],
        rnn_dropout=args['model']['rnn_dropout'],
        input_dropout=args['model']['input_network']['input_layer_dropout'],
        n_layers=args['model']['n_layers'],
        patch_size=args['model']['patch_size'],
        patch_stride=args['model']['patch_stride'],
        a2t_model_id=args['model']['a2t_model_id'],
        device=device,
        use_quantization=args['model'].get('use_quantization', False),
        quantization_bits=args['model'].get('quantization_bits', 8),
        alpha=args['alpha'],
        beta=args['beta'],
        cache_dir=args.get('cache_dir', 'cache/audio_embeddings'),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if 'brain_encoder_state_dict' in checkpoint:
        model.brain_encoder.load_state_dict(checkpoint['brain_encoder_state_dict'])
        print("Loaded checkpoint (brain_encoder)")
    elif 'model_state_dict' in checkpoint:
        from collections import OrderedDict
        brain_encoder_state = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('brain_encoder.'):
                new_key = k.replace('brain_encoder.', '').replace('module.', '')
                brain_encoder_state[new_key] = v
        model.brain_encoder.load_state_dict(brain_encoder_state)
        print("Loaded checkpoint (extracted brain_encoder)")
    else:
        raise ValueError("Checkpoint format not recognized")

    model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model


def evaluate(model, dataloader, dataset, args, device):
    """Evaluate model and return results."""
    model.eval()

    # Map day_idx → session name
    day_to_session = {}
    for day_idx, info in dataset.trial_indicies.items():
        session_path = info['session_path']
        # Extract session name (e.g., 't15.2023.08.11')
        session = [s for s in session_path.split('/') if (s.startswith('t15.20') or s.startswith('t12.20'))][0]
        day_to_session[day_idx] = session

    results = {
        'session': [],
        'day_idx': [],
        'block_num': [],
        'trial_num': [],
        'target_text': [],
        'predicted_text': [],
        'wer': [],
        'num_words': [],
        'edit_distance': [],
    }

    all_predicted = []
    all_target = []

    print(f"\nEvaluating {len(dataloader)} batches...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Load batch
            features = batch['input_features'].to(device, non_blocking=True)
            n_time_steps = batch['n_time_steps'].to(device, non_blocking=True)
            day_indicies = batch['day_indicies'].to(device, non_blocking=True)
            block_nums = batch['block_nums']
            trial_nums = batch['trial_nums']

            # Decode transcriptions
            transcriptions_raw = batch['transcriptions']
            target_texts = []
            for j in range(transcriptions_raw.shape[0]):
                # Decode using chr() to match precompute_all_embeddings.py
                text = ''.join([chr(c) for c in transcriptions_raw[j].cpu().numpy() if c != 0]).strip()
                target_texts.append(text)

            # Apply smoothing (no augmentation)
            with torch.autocast(device_type="cuda", enabled=args['use_amp'], dtype=torch.bfloat16):
                features, n_time_steps = apply_data_augmentations(
                    features, n_time_steps, mode='val',
                    transform_args=args['dataset']['data_transforms'],
                    device=device
                )

                # Generate predictions
                predicted_texts = model.generate(features, day_indicies, max_length=40)

            # Compute per-sample metrics
            for i in range(len(target_texts)):
                pred = predicted_texts[i]
                target = target_texts[i]

                pred_words = pred.lower().strip().split()
                target_words = target.lower().strip().split()
                ed = editdistance.eval(target_words, pred_words)
                wer_sample = ed / len(target_words) if len(target_words) > 0 else 0.0

                day_idx = day_indicies[i].item()
                session = day_to_session.get(day_idx, f'day_{day_idx}')

                results['session'].append(session)
                results['day_idx'].append(day_idx)
                results['block_num'].append(block_nums[i].item())
                results['trial_num'].append(trial_nums[i].item())
                results['target_text'].append(target)
                results['predicted_text'].append(pred)
                results['wer'].append(wer_sample)
                results['num_words'].append(len(target_words))
                results['edit_distance'].append(ed)

                all_predicted.append(pred)
                all_target.append(target)

    # Compute aggregate metrics
    wer, total_ed, total_words = compute_wer(all_predicted, all_target)
    cer, total_ed_chars, total_chars = compute_cer(all_predicted, all_target)

    metrics = {
        'wer': wer,
        'cer': cer,
        'total_samples': len(all_predicted),
        'total_words': total_words,
        'total_chars': total_chars,
        'total_edit_distance': total_ed,
        'total_edit_distance_chars': total_ed_chars,
    }

    # Per-session metrics
    session_data = defaultdict(lambda: {'predicted': [], 'target': []})
    for pred, target, session in zip(all_predicted, all_target, results['session']):
        session_data[session]['predicted'].append(pred)
        session_data[session]['target'].append(target)

    metrics['per_session'] = {}
    for session, data in session_data.items():
        wer_s, ed_s, words_s = compute_wer(data['predicted'], data['target'])
        cer_s, ed_chars_s, chars_s = compute_cer(data['predicted'], data['target'])
        metrics['per_session'][session] = {
            'wer': wer_s,
            'cer': cer_s,
            'n_samples': len(data['predicted']),
            'total_words': words_s,
            'total_edit_distance': ed_s,
        }

    return results, metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Brain-to-Text model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., checkpoint/best_checkpoint)')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                        help='Dataset split to evaluate')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file (default: checkpoint_dir/args.yaml)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: checkpoint_dir)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    args_cmd = parser.parse_args()

    # Auto-detect config if not provided
    checkpoint_dir = os.path.dirname(args_cmd.checkpoint)
    if args_cmd.config is None:
        args_cmd.config = os.path.join(checkpoint_dir, 'args.yaml')
        if not os.path.exists(args_cmd.config):
            print(f"ERROR: Config not found at {args_cmd.config}")
            print("Please provide --config path explicitly")
            sys.exit(1)

    print(f"Loading config from: {args_cmd.config}")
    args = OmegaConf.load(args_cmd.config)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args_cmd.gpu}')
        print(f'Using GPU: {torch.cuda.get_device_name(args_cmd.gpu)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Load model
    model = load_model(args, args_cmd.checkpoint, device)

    # Load dataset
    print(f"\nLoading {args_cmd.split} dataset...")
    data_file = f'data_{args_cmd.split}.hdf5'
    file_paths = [os.path.join(args['dataset']['dataset_dir'], s, data_file)
                 for s in args['dataset']['sessions']]

    # Get all trials for evaluation
    _, eval_trials = train_test_split_indicies(
        file_paths=file_paths,
        test_percentage=1.0,
        seed=args['dataset']['seed'],
        bad_trials_dict=None,
    )

    dataset = BrainToTextDataset(
        trial_indicies=eval_trials,
        split='test',
        days_per_batch=None,
        n_batches=None,
        batch_size=args['dataset']['batch_size'],
        must_include_days=None,
        random_seed=args['dataset']['seed'],
        feature_subset=args['dataset'].get('feature_subset', None)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Loaded {len(dataset)} batches")

    # Run evaluation
    results, metrics = evaluate(model, dataloader, dataset, args, device)

    # Print summary
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS - {args_cmd.split.upper()} SET")
    print("="*80)
    print(f"Total samples:      {metrics['total_samples']}")
    print(f"Total words:        {metrics['total_words']}")
    print(f"\nWord Error Rate:    {metrics['wer']*100:.2f}%")
    print(f"  Edit distance:    {metrics['total_edit_distance']} / {metrics['total_words']}")
    print(f"\nChar Error Rate:    {metrics['cer']*100:.2f}%")
    print(f"  Edit distance:    {metrics['total_edit_distance_chars']} / {metrics['total_chars']}")
    print("\nPer-Session WER:")
    for session, sm in sorted(metrics['per_session'].items()):
        print(f"  {session}: {sm['wer']*100:6.2f}% (n={sm['n_samples']})")
    print("="*80)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args_cmd.output_dir or checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV with per-sample results
    results_csv = os.path.join(output_dir, f'results_{args_cmd.split}_{timestamp}.csv')
    pd.DataFrame(results).to_csv(results_csv, index=False)
    print(f"\nSaved results to: {results_csv}")

    # Save JSON with aggregate metrics
    summary_json = os.path.join(output_dir, f'summary_{args_cmd.split}_{timestamp}.json')
    with open(summary_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {summary_json}")

    # Save TXT summary
    summary_txt = os.path.join(output_dir, f'summary_{args_cmd.split}_{timestamp}.txt')
    with open(summary_txt, 'w') as f:
        f.write(f"Brain-to-Text Model Evaluation - {args_cmd.split.upper()}\n")
        f.write(f"Checkpoint: {args_cmd.checkpoint}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("="*80 + "\n")
        f.write("AGGREGATE METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Total samples:      {metrics['total_samples']}\n")
        f.write(f"Total words:        {metrics['total_words']}\n")
        f.write(f"\nWord Error Rate:    {metrics['wer']*100:.2f}%\n")
        f.write(f"  Edit distance:    {metrics['total_edit_distance']} / {metrics['total_words']}\n")
        f.write(f"\nChar Error Rate:    {metrics['cer']*100:.2f}%\n")
        f.write(f"  Edit distance:    {metrics['total_edit_distance_chars']} / {metrics['total_chars']}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("PER-SESSION METRICS\n")
        f.write("="*80 + "\n")
        for session, sm in sorted(metrics['per_session'].items()):
            f.write(f"{session}: WER={sm['wer']*100:6.2f}% CER={sm['cer']*100:6.2f}% "
                   f"(n={sm['n_samples']}, words={sm['total_words']})\n")
    print(f"Saved summary to: {summary_txt}")

    print("\nDone!")


if __name__ == "__main__":
    main()
