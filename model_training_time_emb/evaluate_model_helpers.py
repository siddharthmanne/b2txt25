import torch
import numpy as np
import h5py
import math

from data_augmentations import gauss_smooth
from dataset import get_sinusoidal_time_embedding, session_to_day_index

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',
]

def load_h5py_file(file_path, b2txt_csv_df):
    data = {
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
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            # sentence_label is a string attribute, not an array - do not use [:]
            sentence_label = g.attrs['sentence_label'] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            # match this trial up with the csv to get the corpus name
            try:
                year, month, day = session.split('.')[1:]
                date = f'{year}-{month}-{day}'
                row = b2txt_csv_df[(b2txt_csv_df['Date'] == date) & (b2txt_csv_df['Block number'] == block_num)]
                corpus_name = row['Corpus'].values[0] if len(row) > 0 else 'Unknown'
            except (IndexError, KeyError):
                corpus_name = 'Unknown'

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
            data['corpus'].append(corpus_name)
    return data

# single decoding step function.
# smooths data and puts it through the model.
def runSingleDecodingStep(x, model, model_args, device, session=None):

    # Use autocast for efficiency
    with torch.autocast(device_type = "cuda", enabled = model_args['use_amp'], dtype = torch.bfloat16):

        # Add time embeddings if configured (BEFORE smoothing, like in dataset)
        time_embedding_dim = model_args['dataset'].get('time_embedding_dim', 0)
        if time_embedding_dim > 0 and session is not None:
            # Get day index for this session
            day_index = session_to_day_index(session)

            # Generate time embedding
            time_emb = get_sinusoidal_time_embedding(day_index, time_embedding_dim)

            # x has shape [batch_size, num_time_steps, feature_dim]
            # Expand time_emb to match: [batch_size, num_time_steps, time_embedding_dim]
            batch_size, num_time_steps, feature_dim = x.shape
            time_emb_expanded = time_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, num_time_steps, -1)

            # Move to same device and dtype as x
            time_emb_expanded = time_emb_expanded.to(x.device, dtype=x.dtype)

            # Concatenate time embeddings to features along last dimension
            x = torch.cat([x, time_emb_expanded], dim=-1)

        x = gauss_smooth(
            inputs = x,
            device = device,
            smooth_kernel_std = model_args['dataset']['data_transforms']['smooth_kernel_std'],
            smooth_kernel_size = model_args['dataset']['data_transforms']['smooth_kernel_size'],
            padding = 'valid',
        )

        with torch.no_grad():
            logits, _ = model(
                x = x,
                states = None, # no initial states
                return_state = True,
            )

    # convert logits from bfloat16 to float32
    logits = logits.float().cpu().numpy()

    return logits