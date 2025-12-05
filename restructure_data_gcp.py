import h5py
import os
import numpy as np
import shutil
from tqdm import tqdm

# Configuration for GCP
DATA_DIR = '/home/Siddharth/data/hdf5_data_final'
OUTPUT_DIR = '/home/Siddharth/data/hdf5_data_restructured'
NUM_SESSIONS = 20
SEED = 42

# 80/10/10 split
TRAIN_PCT = 0.80
VAL_PCT = 0.10
TEST_PCT = 0.10

def main():
    np.random.seed(SEED)

    # Get first 20 sessions
    sessions = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('t15.20')])[:NUM_SESSIONS]
    print(f"Using first {NUM_SESSIONS} sessions")

    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        response = input(f"{OUTPUT_DIR} exists. Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(OUTPUT_DIR)
        else:
            print("Aborted")
            return

    os.makedirs(OUTPUT_DIR)

    # Collect all trials from all sessions (data_train.hdf5 + data_val.hdf5)
    all_trials = []
    for session in sessions:
        # Get trials from data_train.hdf5
        train_path = os.path.join(DATA_DIR, session, 'data_train.hdf5')
        with h5py.File(train_path, 'r') as f:
            for trial_idx in range(len(f.keys())):
                all_trials.append({
                    'session': session,
                    'trial_idx': trial_idx,
                    'source_path': train_path
                })

        # Get trials from data_val.hdf5 if exists
        val_path = os.path.join(DATA_DIR, session, 'data_val.hdf5')
        if os.path.exists(val_path):
            with h5py.File(val_path, 'r') as f:
                for trial_idx in range(len(f.keys())):
                    all_trials.append({
                        'session': session,
                        'trial_idx': trial_idx,
                        'source_path': val_path
                    })

    print(f"Total trials collected: {len(all_trials)}")

    # Shuffle and split 80/10/10
    np.random.shuffle(all_trials)

    n_total = len(all_trials)
    n_train = int(n_total * TRAIN_PCT)
    n_val = int(n_total * VAL_PCT)
    n_test = n_total - n_train - n_val

    train_trials = all_trials[:n_train]
    val_trials = all_trials[n_train:n_train+n_val]
    test_trials = all_trials[n_train+n_val:]

    print(f"\nSplit (80/10/10):")
    print(f"  Train: {len(train_trials)} ({100*len(train_trials)/n_total:.1f}%)")
    print(f"  Val:   {len(val_trials)} ({100*len(val_trials)/n_total:.1f}%)")
    print(f"  Test:  {len(test_trials)} ({100*len(test_trials)/n_total:.1f}%)")
    print(f"  Total: {n_total}")

    # Create new directory structure: 20 subfolders, each with data_train.hdf5, data_val.hdf5, data_test.hdf5
    for session in tqdm(sessions, desc="Processing sessions"):
        session_dir = os.path.join(OUTPUT_DIR, session)
        os.makedirs(session_dir, exist_ok=True)

        # Filter trials by session
        session_train = [t for t in train_trials if t['session'] == session]
        session_val = [t for t in val_trials if t['session'] == session]
        session_test = [t for t in test_trials if t['session'] == session]

        # Create HDF5 files
        create_hdf5(session_train, os.path.join(session_dir, 'data_train.hdf5'))
        create_hdf5(session_val, os.path.join(session_dir, 'data_val.hdf5'))
        create_hdf5(session_test, os.path.join(session_dir, 'data_test.hdf5'))

    print(f"\nData restructured in: {OUTPUT_DIR}")
    print(f"Structure: {NUM_SESSIONS} session folders × 3 files (data_train.hdf5, data_val.hdf5, data_test.hdf5)")

def create_hdf5(trials, output_path):
    """Create HDF5 file with specified trials"""
    with h5py.File(output_path, 'w') as out_f:
        for new_idx, trial_info in enumerate(trials):
            source_path = trial_info['source_path']
            trial_idx = trial_info['trial_idx']

            with h5py.File(source_path, 'r') as in_f:
                old_key = f'trial_{trial_idx:04d}'
                new_key = f'trial_{new_idx:04d}'
                in_f.copy(old_key, out_f, new_key)

if __name__ == '__main__':
    main()
