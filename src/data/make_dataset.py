import numpy as np
import os
import argparse
import pandas as pd
import wfdb
import pyedflib
import json
from datetime import datetime

def write_edf(output_path, signals, fs, annotations, header_info=None):
    """
    Writes signals and annotations to an EDF+ file using pyedflib.
    
    Args:
        output_path: Path to save the EDF file.
        signals: List of numpy arrays (one per channel). All must have same length.
        fs: Sampling frequency (int).
        annotations: List of tuples (onset, duration, description).
        header_info: Dict with patient info (optional).
    """
    n_channels = len(signals)
    with pyedflib.EdfWriter(output_path, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        
        # Set channel headers
        channel_info = []
        for i in range(n_channels):
            ch_dict = {
                'label': f"Lead_{i+1}", 
                'dimension': 'mV', 
                'sample_frequency': fs, 
                'physical_min': np.min(signals), 
                'physical_max': np.max(signals), 
                'digital_min': -32768, 
                'digital_max': 32767, 
                'transducer': 'Electrode', 
                'prefilter': ''
            }
            channel_info.append(ch_dict)
        
        f.setSignalHeaders(channel_info)
        
        # Set file header
        if header_info:
            f.setPatientCode(header_info.get('patient_code', ''))
            f.setRecordingAdditional(header_info.get('recording_info', ''))
            if 'startdate' in header_info:
                f.setStartdatetime(header_info['startdate'])
        
        # Write signals
        f.writeSamples(signals)
        
        # Write annotations
        for onset, duration, desc in annotations:
            f.writeAnnotation(onset, duration, desc)

def process_ptbxl_data(input_dir, output_dir, fs=100):
    """
    Reads PTB-XL data and converts it into 3 EDF+ files (train, val, test).
    
    Args:
        input_dir: Path to the root of the PTB-XL dataset.
        output_dir: Path where EDF files will be saved.
        fs: Sampling rate to use (100 or 500). Default 100.
    """
    
    # Check input
    database_path = os.path.join(input_dir, 'ptbxl_database.csv')
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Could not find {database_path}. Is the input directory correct?")

    print(f"Loading metadata from {database_path}...")
    df = pd.read_csv(database_path, index_col='ecg_id')
    
    # Define splits
    # 1-8: Train, 9: Val, 10: Test
    splits = {
        'train': df[df['strat_fold'] <= 8],
        'val': df[df['strat_fold'] == 9],
        'test': df[df['strat_fold'] == 10]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_df in splits.items():
        print(f"Processing {split_name} set ({len(split_df)} records)...")
        
        all_signals = [] # List to hold all concatenated signals for each channel
        annotations = []
        current_time = 0.0
        
        # Initialize list of lists for 12 leads
        # Typically PTB-XL has 12 leads. We'll verify on first read.
        initialized = False
        n_leads = 12
        channel_buffers = [[] for _ in range(n_leads)]
        
        for ecg_id, row in split_df.iterrows():
            # Construct path to signal file
            # filename_lr usually looks like 'records100/00000/00001_lr'
            # We need to handle fs choice. 
            # If fs=100 use filename_lr, if fs=500 use filename_hr
            rel_path = row['filename_lr'] if fs == 100 else row['filename_hr']
            record_path = os.path.join(input_dir, rel_path)
            
            try:
                # Read signal
                # channels usually: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
                signals, fields = wfdb.rdsamp(record_path)
                
                if not initialized:
                    n_leads = signals.shape[1]
                    channel_buffers = [[] for _ in range(n_leads)]
                    initialized = True
                
                # Reshape: (samples, channels) -> list of (samples,)
                for i in range(n_leads):
                    channel_buffers[i].append(signals[:, i])
                
                # Create annotation
                # Duration is usually 10s
                duration = signals.shape[0] / fields['fs']
                
                # Description JSON
                desc = {
                    'ecg_id': ecg_id,
                    'diagnostic_superclass': row['diagnostic_superclass'] if 'diagnostic_superclass' in row else []
                }
                # Sanitize for annotation text (must be modest length)
                # usage: json string or simple format
                # We'll use a simplified string for compatibility: "ID:<id> SCF:<json_list>"
                # But JSON is cleaner if reader supports it. Let's keep it simple key-value string.
                # Actually, EDF+ supports UTF8 but length is finite.
                # Let's save crucial info.
                
                # desc_str = json.dumps(desc) # might be too long?
                # Let's just store the ID. The loader can look up metadata from CSV if needed using ID.
                # Or store the diagnostic class.
                classes = row.get('scp_codes', '{}')
                desc_str = f"ID:{ecg_id} L:{classes}"
                
                annotations.append((current_time, duration, desc_str))
                
                current_time += duration
                
            except Exception as e:
                print(f"Error reading record {ecg_id} at {record_path}: {e}")
                continue

        # Concatenate buffers
        print(f"Concatenating signals for {split_name}...")
        final_signals = []
        for i in range(n_leads):
            final_signals.append(np.concatenate(channel_buffers[i]))
            
        # Write to EDF
        output_file = os.path.join(output_dir, f"{split_name}.edf")
        print(f"Writing to {output_file}...")
        write_edf(output_file, final_signals, fs, annotations, header_info={'patient_code': f"PTB-XL {split_name}"})
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PTB-XL to EDF+ files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory of PTB-XL dataset.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for EDF files.")
    parser.add_argument("--fs", type=int, default=100, choices=[100, 500], help="Sampling rate to use (100 or 500).")
    
    args = parser.parse_args()
    
    process_ptbxl_data(args.input_dir, args.output_dir, args.fs)
