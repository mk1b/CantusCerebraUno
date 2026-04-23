import json
import os
import pickle
import numpy as np
from natsort import natsorted
from collections import defaultdict
import random
import gc

# --- CONFIGURATION ---
# Correct paths based on your previous messages
data_folder = os.path.expanduser('~/CantusCerebra/processed_data/processed_siena/data_generate')
output_dir = os.path.expanduser('~/CantusCerebra/processed_data/processed_siena/json_generate')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

save_folder_train = os.path.join(output_dir, 'train.json')
save_folder_val = os.path.join(output_dir, 'val.json')
save_folder_test = os.path.join(output_dir, 'test.json')

# Base parameters
sampling_rate = 512
ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5', 'Cp1', 'Cp5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2',
            'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10']
num_channels = len(ch_names)
random.seed(42)

def load_subject_metadata(subject_folder):
    """
    Loads ONLY metadata (paths and labels). 
    Crucial: Does NOT store the heavy 'X' signal in the list to save RAM.
    """
    subject_metadata = []
    subject_num = int(os.path.basename(subject_folder)[2:])
    subject_name = f"PN{subject_num:02d}"

    # Get all .pkl files in order
    files = natsorted([f for f in os.listdir(subject_folder) if f.endswith('.pkl')])
    
    for file in files:
        file_path = os.path.join(subject_folder, file)
        try:
            # We open pickle ONLY to get the Label (Y)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            subject_metadata.append({
                "subject_id": subject_num,
                "subject_name": subject_name,
                "file": file_path,
                "label": int(data['Y'])
                # NOTE: 'eeg_data' is intentionally NOT added here.
            })
            
            # Explicitly delete to free RAM immediately
            del data
            
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            
    return subject_metadata

def compute_stats_streaming(metadata_list):
    """
    Computes global normalization stats by loading files ONE BY ONE.
    Uses 'Running Sum' algorithm to avoid holding data in RAM.
    """
    print(f"Computing statistics on {len(metadata_list)} files...")
    
    # Accumulators
    sum_x = np.zeros(num_channels)
    sum_sq_x = np.zeros(num_channels)
    total_count = 0
    
    global_max = -np.inf
    global_min = np.inf
    
    for idx, item in enumerate(metadata_list):
        try:
            # 1. Load the heavy data temporarily
            with open(item['file'], 'rb') as f:
                data = pickle.load(f)
            
            X = data['X'] # Shape: (Channels, Time)
            
            # 2. Update Stats
            # Ensure shape is (Channels, Time)
            if X.shape[0] != num_channels:
                if X.shape[1] == num_channels: X = X.T

            num_samples = X.shape[1]
            
            # Update Min/Max
            current_max = np.max(X)
            current_min = np.min(X)
            if current_max > global_max: global_max = current_max
            if current_min < global_min: global_min = current_min
            
            # Update Sums (Sum across time dimension)
            sum_x += np.sum(X, axis=1)
            sum_sq_x += np.sum(X**2, axis=1)
            total_count += num_samples
            
            # 3. Free Memory IMMEDIATELY
            del data, X
            
            # Periodic print
            if (idx + 1) % 500 == 0:
                print(f"Processed stats for {idx + 1}/{len(metadata_list)} files...")

        except Exception as e:
            print(f"Skipping stats for {item['file']}: {e}")

    # Final Calculation
    mean = sum_x / total_count
    
    # Variance = E[X^2] - (E[X])^2
    mean_sq = sum_sq_x / total_count
    var = mean_sq - (mean ** 2)
    std = np.sqrt(var) # Standard Deviation

    print("Stats computation complete.")
    return mean.tolist(), std.tolist(), float(global_max), float(global_min)

def split_subject_data(subject_data, val_ratio=0.2):
    """Splits metadata into train/val."""
    label_to_data = defaultdict(list)
    for data in subject_data:
        label_to_data[data["label"]].append(data)

    train_data, val_data = [], []
    for label, data_list in label_to_data.items():
        random.shuffle(data_list)
        split_idx = int(len(data_list) * (1 - val_ratio))
        train_data.extend(data_list[:split_idx])
        val_data.extend(data_list[split_idx:])

    return train_data, val_data

def save_dataset(data_list, save_path, norm_params):
    """Saves the lightweight metadata list to JSON."""
    mean, std, max_val, min_val = norm_params
    
    dataset = {
        "subject_data": data_list, 
        "dataset_info": {
            "sampling_rate": sampling_rate,
            "ch_names": ch_names,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "std": std
        }
    }

    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {save_path}")

def main():
    # 1. Gather all subject folders
    print(f"Scanning subject folders in {data_folder}...")
    
    if not os.path.exists(data_folder):
        print(f"ERROR: Data folder not found at {data_folder}")
        return

    subject_folders = natsorted(
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.startswith("PN") and os.path.isdir(os.path.join(data_folder, f))
    )

    if not subject_folders:
        print("No subject folders (PN*) found.")
        return

    # Split subjects by ID (<=12 is Train, >12 is Test)
    train_subjects = [s for s in subject_folders if int(os.path.basename(s)[2:]) <= 12]
    test_subjects = [s for s in subject_folders if int(os.path.basename(s)[2:]) > 12]

    # 2. PASS 1: Metadata Loading & Splitting
    print("--- PASS 1: Loading Metadata (No Signals) ---")
    
    all_train_data, all_val_data = [], []
    for subject in train_subjects:
        print(f"Loading metadata: {os.path.basename(subject)}")
        subj_meta = load_subject_metadata(subject)
        t_data, v_data = split_subject_data(subj_meta)
        all_train_data.extend(t_data)
        all_val_data.extend(v_data)

    all_test_data = []
    for subject in test_subjects:
        print(f"Loading metadata: {os.path.basename(subject)}")
        all_test_data.extend(load_subject_metadata(subject))

    print(f"\nMetadata Loaded.")
    print(f"Train samples: {len(all_train_data)}")
    print(f"Val samples:   {len(all_val_data)}")
    print(f"Test samples:  {len(all_test_data)}")

    # 3. PASS 2: Streaming Statistics (CPU Heavy, RAM Light)
    # Only compute stats on TRAINING set to avoid leakage
    print("\n--- PASS 2: Computing Normalization Stats ---")
    if len(all_train_data) > 0:
        norm_params = compute_stats_streaming(all_train_data)
    else:
        print("Warning: No training data found. Using dummy stats.")
        norm_params = ([0]*num_channels, [1]*num_channels, 1, -1)

    # 4. Save
    print("\n--- Saving JSONs ---")
    save_dataset(all_train_data, save_folder_train, norm_params)
    save_dataset(all_val_data, save_folder_val, norm_params)
    save_dataset(all_test_data, save_folder_test, norm_params)

if __name__ == "__main__":
    main()
