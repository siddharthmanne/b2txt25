import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from rnn_model import GRUDecoder
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep, LOGIT_TO_PHONEME

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')
parser.add_argument('--config_path', type=str, required=True,
                    help='Path to the YAML config file (e.g., rnn_args.yaml).')
args = parser.parse_args()

# load config
config = OmegaConf.load(args.config_path)

# extract paths and settings from config
model_path = config.get('model_path', '../data/t15_pretrained_rnn_baseline')
data_dir = config['dataset']['dataset_dir']
csv_path = config.get('csv_path', '../../data/t15_copyTaskData_description.csv')
gpu_number = int(config.get('gpu_number', 1))
output_dir = config.get('eval_output_dir', 'eval_results')

# get eval dataset from dataset section
eval_days = config['dataset'].get('eval_dataset', None)
eval_days_probability_val = config['dataset'].get('eval_dataset_probability_val', None)

if eval_days is None:
    raise ValueError('eval_dataset not found in config file. Please add dataset.eval_dataset list.')
if eval_days_probability_val is None:
    raise ValueError('eval_dataset_probability_val not found in config file. Please add dataset.eval_dataset_probability_val list.')
if len(eval_days) != len(eval_days_probability_val):
    raise ValueError('eval_dataset and eval_dataset_probability_val must have the same length.')

print(f'Model path: {model_path}')
print(f'Data directory: {data_dir}')
print(f'Evaluating on sessions: {eval_days}')
print(f'Val set availability: {eval_days_probability_val}')
print(f'Output directory: {output_dir}')
print()

# load csv file
b2txt_csv_df = pd.read_csv(csv_path)

# load model args
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# set up gpu device
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# define model
model = GRUDecoder(
    neural_dim = model_args['model']['n_input_features'],
    n_units = model_args['model']['n_units'],
    n_classes = model_args['dataset']['n_classes'],
    rnn_dropout = model_args['model']['rnn_dropout'],
    input_dropout = model_args['model']['input_network']['input_layer_dropout'],
    n_layers = model_args['model']['n_layers'],
    patch_size = model_args['model']['patch_size'],
    patch_stride = model_args['model']['patch_stride'],
)

# load model weights
checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False)
# rename keys to not start with "module." (happens if model was saved with DataParallel)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])  

# add model to device
model.to(device) 

# set model to eval mode
model.eval()

# load data for each session (train + val if available)
test_data = {}
total_test_trials = 0

for i, session in enumerate(eval_days):
    session_data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
        'corpus': [],
        'logits': [],
        'pred_seq': [],
    }

    # load train data
    train_file = os.path.join(data_dir, session, 'data_train.hdf5')
    if os.path.exists(train_file):
        train_data = load_h5py_file(train_file, b2txt_csv_df)
        for key in ['neural_features', 'n_time_steps', 'seq_class_ids', 'seq_len',
                    'transcriptions', 'sentence_label', 'session', 'block_num', 'trial_num', 'corpus']:
            if key in train_data and key in session_data:
                session_data[key].extend(train_data[key])
        print(f'Loaded {len(train_data["neural_features"])} train trials for session {session}.')

    # load val data if available
    if eval_days_probability_val[i] == 1:
        val_file = os.path.join(data_dir, session, 'data_val.hdf5')
        if os.path.exists(val_file):
            val_data = load_h5py_file(val_file, b2txt_csv_df)
            for key in ['neural_features', 'n_time_steps', 'seq_class_ids', 'seq_len',
                        'transcriptions', 'sentence_label', 'session', 'block_num', 'trial_num', 'corpus']:
                if key in val_data and key in session_data:
                    session_data[key].extend(val_data[key])
            print(f'Loaded {len(val_data["neural_features"])} val trials for session {session}.')

    test_data[session] = session_data
    total_test_trials += len(session_data["neural_features"])

print(f'Total number of trials across all sessions: {total_test_trials}')
print()


# put neural data through the pretrained model to get phoneme predictions (logits)
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # run decoding step
            logits = runSingleDecodingStep(neural_input, model, model_args, device, session=session)
            data['logits'].append(logits)

            pbar.update(1)
pbar.close()


# convert logits to phoneme sequences and print them out
for session, data in test_data.items():
    data['pred_seq'] = []
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        pred_seq = np.argmax(logits, axis=-1)
        # remove blanks (0)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # remove consecutive duplicates
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        # convert to phonemes
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        # add to data
        data['pred_seq'].append(pred_seq)

        # print out the predicted sequences
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        sentence_label = data['sentence_label'][trial]
        true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
        true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]

        print(f'Sentence label:      {sentence_label}')
        print(f'True sequence:       {" ".join(true_seq)}')
        print(f'Predicted Sequence:  {" ".join(pred_seq)}')
        print()


# calculate the aggregate phoneme error rate (PER)
# this matches the metric used during training validation
print('='*80)
print('Calculating Phoneme Error Rate (PER)')
print('='*80)
print()

per_results = {
    'session': [],
    'block': [],
    'trial': [],
    'sentence_label': [],
    'true_phonemes': [],
    'pred_phonemes': [],
    'edit_distance': [],
    'num_phonemes': [],
}

total_true_length = 0
total_edit_distance = 0

for session, data in test_data.items():
    for trial in range(len(data['pred_seq'])):
        # get true and predicted phoneme sequences
        true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
        true_phonemes = [LOGIT_TO_PHONEME[p] for p in true_seq]
        pred_phonemes = data['pred_seq'][trial]

        # calculate edit distance between phoneme sequences
        ed = editdistance.eval(true_phonemes, pred_phonemes)

        total_true_length += len(true_phonemes)
        total_edit_distance += ed

        # store results
        per_results['session'].append(session)
        per_results['block'].append(data['block_num'][trial])
        per_results['trial'].append(data['trial_num'][trial])
        per_results['sentence_label'].append(data['sentence_label'][trial])
        per_results['true_phonemes'].append(true_phonemes)
        per_results['pred_phonemes'].append(pred_phonemes)
        per_results['edit_distance'].append(ed)
        per_results['num_phonemes'].append(len(true_phonemes))

        # print individual trial results
        print(f'{session} - Block {data["block_num"][trial]}, Trial {data["trial_num"][trial]}')
        print(f'Sentence label:      {data["sentence_label"][trial]}')
        print(f'True phonemes:       {" ".join(true_phonemes)}')
        print(f'Predicted phonemes:  {" ".join(pred_phonemes)}')
        if len(true_phonemes) > 0:
            print(f'PER: {ed} / {len(true_phonemes)} = {100 * ed / len(true_phonemes):.2f}%')
        else:
            print(f'PER: N/A (no true phonemes)')
        print()

print(f'Total true phoneme sequence length: {total_true_length}')
print(f'Total edit distance: {total_edit_distance}')
if total_true_length > 0:
    print(f'Aggregate Phoneme Error Rate (PER): {100 * total_edit_distance / total_true_length:.2f}%')
else:
    print(f'Aggregate Phoneme Error Rate (PER): N/A (no phonemes)')
print()

# Calculate per-day metrics
print('='*80)
print('Per-Day Metrics:')
print('='*80)

per_day_metrics = {}
for session in eval_days:
    # Filter results for this session
    day_true_length = 0
    day_edit_distance = 0
    day_trial_count = 0

    for i in range(len(per_results['session'])):
        if per_results['session'][i] == session:
            day_true_length += per_results['num_phonemes'][i]
            day_edit_distance += per_results['edit_distance'][i]
            day_trial_count += 1

    if day_true_length > 0:
        day_per = 100 * day_edit_distance / day_true_length
    else:
        day_per = 0.0

    per_day_metrics[session] = {
        'per': day_per,
        'edit_distance': day_edit_distance,
        'num_phonemes': day_true_length,
        'num_trials': day_trial_count,
    }

    print(f'{session}:')
    print(f'  Trials: {day_trial_count}')
    print(f'  Total phonemes: {day_true_length}')
    print(f'  Total edit distance: {day_edit_distance}')
    print(f'  PER: {day_per:.2f}%')
    print()

# Calculate average PER across days
avg_per = np.mean([metrics['per'] for metrics in per_day_metrics.values()])
print(f'Average PER across {len(eval_days)} days: {avg_per:.2f}%')
print('='*80)
print()

# Save results to eval.txt
os.makedirs(output_dir, exist_ok=True)
eval_output_file = os.path.join(output_dir, 'eval.txt')

with open(eval_output_file, 'w') as f:
    f.write('='*80 + '\n')
    f.write('Evaluation Results\n')
    f.write('='*80 + '\n')
    f.write(f'Model path: {model_path}\n')
    f.write(f'Data directory: {data_dir}\n')
    f.write(f'Config path: {args.config_path}\n')
    f.write(f'Evaluated sessions: {eval_days}\n')
    f.write('\n')

    f.write('='*80 + '\n')
    f.write('Overall Metrics:\n')
    f.write('='*80 + '\n')
    f.write(f'Total trials: {len(per_results["pred_phonemes"])}\n')
    f.write(f'Total phonemes: {total_true_length}\n')
    f.write(f'Total edit distance: {total_edit_distance}\n')
    if total_true_length > 0:
        f.write(f'Overall PER: {100 * total_edit_distance / total_true_length:.2f}%\n')
    else:
        f.write(f'Overall PER: N/A (no phonemes)\n')
    f.write('\n')

    f.write('='*80 + '\n')
    f.write('Per-Day Metrics:\n')
    f.write('='*80 + '\n')

    for session in eval_days:
        metrics = per_day_metrics[session]
        f.write(f'{session}:\n')
        f.write(f'  Trials: {metrics["num_trials"]}\n')
        f.write(f'  Total phonemes: {metrics["num_phonemes"]}\n')
        f.write(f'  Total edit distance: {metrics["edit_distance"]}\n')
        f.write(f'  PER: {metrics["per"]:.2f}%\n')
        f.write('\n')

    f.write('='*80 + '\n')
    f.write(f'Average PER across {len(eval_days)} days: {avg_per:.2f}%\n')
    f.write('='*80 + '\n')

print(f'Evaluation results saved to: {eval_output_file}')
print()


# write predicted phoneme sequences to a csv file. put a timestamp in the filename (YYYYMMDD_HHMMSS)
output_file = os.path.join(output_dir, f'predicted_phonemes_{time.strftime("%Y%m%d_%H%M%S")}.csv')
ids = [i for i in range(len(per_results['pred_phonemes']))]
pred_phoneme_strings = [' '.join(phonemes) for phonemes in per_results['pred_phonemes']]
true_phoneme_strings = [' '.join(phonemes) for phonemes in per_results['true_phonemes']]
df_out = pd.DataFrame({
    'id': ids,
    'session': per_results['session'],
    'block': per_results['block'],
    'trial': per_results['trial'],
    'sentence_label': per_results['sentence_label'],
    'true_phonemes': true_phoneme_strings,
    'pred_phonemes': pred_phoneme_strings,
    'edit_distance': per_results['edit_distance'],
    'num_phonemes': per_results['num_phonemes'],
})
df_out.to_csv(output_file, index=False)
print(f'Predicted phoneme sequences saved to: {output_file}')