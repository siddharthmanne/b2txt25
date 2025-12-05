"""
Test script to verify that validation data filtering is working correctly.
This script checks that OpenWebText and Random corpus blocks are excluded from validation.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset import train_test_split_indicies
import h5py

# Define the sessions to test (same as in rnn_args.yaml)
sessions = [
    't15.2023.08.11',
    't15.2023.08.13',
    't15.2023.08.18',
    't15.2023.08.20',
    't15.2023.08.25',
    't15.2023.08.27',
    't15.2023.09.01',
    't15.2023.09.03',
    't15.2023.09.24',
    't15.2023.09.29',
    't15.2023.10.01',
    't15.2023.10.06',
    't15.2023.10.08',
    't15.2023.10.13',
    't15.2023.10.15',
]

# Define the base path (adjust this to your actual data path)
# You'll need to update this path to match your system
base_path = '/Users/Siddharth/nejm-brain-to-text/data/hdf5_data_final'

# Define validation blocks to exclude (OpenWebText and Random corpus blocks)
# NOTE: Session format uses dots (.), not dashes (-)
exclude_val_blocks_dict = {
    't15.2023.09.29': [4],
    't15.2023.10.01': [8],
    't15.2023.10.08': [11, 12],
    't15.2023.10.13': [6],
    't15.2023.10.15': [10],
}

print("=" * 80)
print("TESTING VALIDATION DATA FILTERING")
print("=" * 80)
print("\nThis test will verify that OpenWebText and Random corpus blocks are excluded")
print("from the validation data.\n")

# Create file paths
val_file_paths = [os.path.join(base_path, s, 'data_val.hdf5') for s in sessions]

# Test WITHOUT filtering
print("=" * 80)
print("TEST 1: Validation data WITHOUT filtering")
print("=" * 80)
_, val_trials_unfiltered = train_test_split_indicies(
    file_paths=val_file_paths,
    test_percentage=1,
    seed=1,
    bad_trials_dict=None,
    exclude_val_blocks_dict=None,  # No filtering
)

total_unfiltered = 0
for day_idx, day_data in val_trials_unfiltered.items():
    num_trials = len(day_data['trials'])
    total_unfiltered += num_trials
    print(f"Day {day_idx} ({sessions[day_idx]}): {num_trials} trials")

print(f"\nTotal validation trials WITHOUT filtering: {total_unfiltered}")

# Test WITH filtering
print("\n" + "=" * 80)
print("TEST 2: Validation data WITH filtering (excluding OpenWebText/Random blocks)")
print("=" * 80)
_, val_trials_filtered = train_test_split_indicies(
    file_paths=val_file_paths,
    test_percentage=1,
    seed=1,
    bad_trials_dict=None,
    exclude_val_blocks_dict=exclude_val_blocks_dict,  # With filtering
)

total_filtered = 0
for day_idx, day_data in val_trials_filtered.items():
    num_trials = len(day_data['trials'])
    total_filtered += num_trials
    print(f"Day {day_idx} ({sessions[day_idx]}): {num_trials} trials")

print(f"\nTotal validation trials WITH filtering: {total_filtered}")

# Show the difference
print("\n" + "=" * 80)
print("FILTERING RESULTS")
print("=" * 80)
print(f"Trials removed by filtering: {total_unfiltered - total_filtered}")
print(f"Percentage of trials kept: {(total_filtered / total_unfiltered * 100):.1f}%")

# Verify that excluded blocks are actually excluded
print("\n" + "=" * 80)
print("VERIFICATION: Checking that excluded blocks are actually filtered out")
print("=" * 80)

for day_idx, day_data in val_trials_filtered.items():
    session = sessions[day_idx]
    session_path = day_data['session_path']

    if not os.path.exists(session_path):
        print(f"⚠️  WARNING: File not found: {session_path}")
        continue

    # Check if this session has blocks to exclude
    session_key = session.replace('.', '-')  # Convert format
    if session_key in exclude_val_blocks_dict:
        excluded_blocks = exclude_val_blocks_dict[session_key]

        # Get all block numbers in the filtered trials
        with h5py.File(session_path, 'r') as f:
            block_nums_in_filtered = []
            for trial_idx in day_data['trials']:
                key = f'trial_{trial_idx:04d}'
                block_num = f[key].attrs['block_num']
                block_nums_in_filtered.append(block_num)

        # Check if any excluded blocks are present
        blocks_found = set(block_nums_in_filtered) & set(excluded_blocks)

        if blocks_found:
            print(f"❌ FAIL: {session} - Excluded blocks {list(blocks_found)} are still present!")
        else:
            print(f"✓ PASS: {session} - Excluded blocks {excluded_blocks} successfully filtered")
            print(f"         Remaining blocks in validation: {sorted(set(block_nums_in_filtered))}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
