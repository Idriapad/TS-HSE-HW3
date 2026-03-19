import os
import random
import pandas as pd
import numpy as np
import config

def process_tsf_to_csv(tsf_path, output_csv, n_samples, seed):
    with open(tsf_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '@data':
            data_start = i + 1
            break
            
    data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
    
    random.seed(seed)
    sampled_lines = random.sample(data_lines, n_samples)
    
    records = []
    
    for line in sampled_lines:
        parts = line.split(':')
        series_name = parts[0]
        
        values_str = parts[-1]
        values = [float(v) if v != '?' else np.nan for v in values_str.split(',')]
        
        dates = pd.date_range(start='2000-01-31', periods=len(values), freq='ME')
        
        for dt, val in zip(dates, values):
            records.append({
                'unique_id': series_name,
                'ds': dt,
                'y': val
            })
            
    df = pd.DataFrame(records)
    
    df.dropna(subset=['y'], inplace=True)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f"Размер итогового датасета: {df.shape}")

if __name__ == "__main__":
    process_tsf_to_csv(
        tsf_path=config.RAW_DATA_PATH,
        output_csv=config.PROCESSED_DATA_PATH,
        n_samples=config.N_SAMPLES,
        seed=config.RANDOM_SEED
    )
