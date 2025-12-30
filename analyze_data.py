import pandas as pd
import os
import glob
import sys
sys.path.insert(0, '.')

# Reload the preprocessing module
import importlib
import token_malice_prediction.data.preprocessing
importlib.reload(token_malice_prediction.data.preprocessing)
from token_malice_prediction.data.preprocessing import TokenPreprocessor

# Test with the fixed preprocessor
preprocessor = TokenPreprocessor(
    data_dir='data_solana',
    malice_threshold=0.9,
    observation_days=30,
    training_window_days=20,
    sudden_drop_threshold=0.8,
    sudden_drop_window_hours=48,
    classification_mode='sudden_drop'
)

csv_files = preprocessor.get_csv_files()[:10]

print("Analyzing with FIXED price calculation:")
print("=" * 60)

malicious_count = 0
benign_count = 0

for f in csv_files:
    df = preprocessor.load_csv(f)
    df = df.sort_values('timestamp')
    
    name = f.stem[:25]
    
    # Get price stats (now using actual token price)
    first_week = df.head(len(df)//4)['Value']  # First 25%
    last_week = df.tail(len(df)//4)['Value']   # Last 25%
    
    price_change = (last_week.mean() - first_week.mean()) / first_week.mean() * 100 if first_week.mean() > 0 else 0
    
    # Check if it passes duration filter
    time_span_days = (df['timestamp'].max() - df['timestamp'].min()) / 86400
    
    if time_span_days >= 30:
        label = preprocessor.compute_label(df)
        label_str = "MALICIOUS" if label == 1 else "BENIGN"
        if label == 1:
            malicious_count += 1
        else:
            benign_count += 1
    else:
        label_str = "SKIP (too short)"
    
    print(f"\n{name}")
    print(f"  Days: {time_span_days:.1f}")
    print(f"  First quarter avg price: ${first_week.mean():.8f}")
    print(f"  Last quarter avg price:  ${last_week.mean():.8f}")
    print(f"  Price change: {price_change:+.1f}%")
    print(f"  Label: {label_str}")

print(f"\n{'=' * 60}")
print(f"Summary: {malicious_count} malicious, {benign_count} benign (out of tokens with 30+ days)")
