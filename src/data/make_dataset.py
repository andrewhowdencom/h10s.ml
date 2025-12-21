import numpy as np
import os
import argparse
import pandas as pd
import wfdb
import pyedflib
import json
from datetime import datetime
from scipy.signal import resample

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

def process_ptbxl_data(input_dir, output_dir, fs=100, target_fs=130):
    """
    Reads PTB-XL data and converts it into 3 EDF+ files (train, val, test).
    
    Args:
        input_dir: Path to the root of the PTB-XL dataset.
        output_dir: Path where EDF files will be saved.
        fs: Original sampling rate of PTB-XL data (100 or 500). Default 100.
        target_fs: Target sampling rate for the output EDF (e.g. 130 for Polar H10).
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
        
        current_time = 0.0
        
        # We only want Lead I (Index 0)
        # We will accumulate a single channel
        channel_buffer = [] 
        annotations = []
        
        for ecg_id, row in split_df.iterrows():
            # Construct path to signal file
            rel_path = row['filename_lr'] if fs == 100 else row['filename_hr']
            record_path = os.path.join(input_dir, rel_path)
            
            try:
                # Read signal
                # channels: I, II, III...
                signals, fields = wfdb.rdsamp(record_path)
                
                # Extract Lead I (Index 0)
                lead_i = signals[:, 0]
                
                # Resample if needed
                if target_fs != fs:
                    # Calculate new number of samples
                    num_samples = int(len(lead_i) * target_fs / fs)
                    lead_i = resample(lead_i, num_samples)
                
                channel_buffer.append(lead_i)
                
                # Create annotation
                # Duration should match the new resampled signal
                duration = len(lead_i) / target_fs
                
                # Description
                classes = row.get('scp_codes', '{}')
                desc_str = f"ID:{ecg_id} L:{classes}"
                
                annotations.append((current_time, duration, desc_str))
                
                current_time += duration
                
            except Exception as e:
                print(f"Error reading record {ecg_id} at {record_path}: {e}")
                continue

        # Concatenate
        print(f"Concatenating signals for {split_name}...")
        final_signal = np.concatenate(channel_buffer)
        
        # Write to EDF
        output_file = os.path.join(output_dir, f"{split_name}.edf")
        print(f"Writing to {output_file} (Lead I, {target_fs}Hz)...")
        # Pass as list of signals (just one)
        write_edf(output_file, [final_signal], target_fs, annotations, header_info={'patient_code': f"PTB-XL {split_name} Lead I"})
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PTB-XL to EDF+ files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory of PTB-XL dataset.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for EDF files.")
    parser.add_argument("--fs", type=int, default=100, choices=[100, 500], help="Original sampling rate (100 or 500).")
    parser.add_argument("--target_fs", type=int, default=130, help="Target sampling rate (default 130 for H10).")
    
    args = parser.parse_args()
    
    process_ptbxl_data(args.input_dir, args.output_dir, args.fs, args.target_fs)
