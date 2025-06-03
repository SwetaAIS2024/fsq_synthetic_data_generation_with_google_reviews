# Synthetic User Generation
# This script generates synthetic users and assigns them plausible check-in histories based on real user behavior.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
n_synth_users = 1000  # Number of synthetic users to generate
random_seed = 42
np.random.seed(random_seed)

# Load real FSQ check-ins
checkins = pd.read_csv('fsq_original_dataset/singapore_checkins.txt', sep='\t', names=['user_id', 'poi_id', 'timestamp', 'misc'])

# Analyze real user behavior
df_user_counts = checkins['user_id'].value_counts()
df_poi_counts = checkins['poi_id'].value_counts()
all_pois = df_poi_counts.index.tolist()
poi_weights = df_poi_counts.values / df_poi_counts.values.sum()

# For timestamp sampling
real_timestamps = pd.to_datetime(checkins['timestamp'], errors='coerce').dropna()
real_timestamps_np = real_timestamps.values  # Convert to numpy array once
time_min, time_max = real_timestamps.min(), real_timestamps.max()

def sample_timestamp():
    # Sample a timestamp from the real distribution, or uniformly if not enough data
    if len(real_timestamps_np) > 0:
        return np.random.choice(real_timestamps_np)
    else:
        delta = time_max - time_min
        rand_seconds = np.random.randint(0, int(delta.total_seconds()))
        return time_min + timedelta(seconds=rand_seconds)

synthetic_checkins = []
for i in range(n_synth_users):
    synth_user_id = f'synth_user_{i+1}'
    # Sample number of check-ins for this user from real user distribution
    n_checkins = df_user_counts.sample(1).values[0]
    # Sample POIs for this user (with replacement, weighted by popularity)
    pois = np.random.choice(all_pois, size=n_checkins, p=poi_weights)
    for poi in pois:
        ts = sample_timestamp()
        synthetic_checkins.append({
            'user_id': synth_user_id,
            'poi_id': poi,
            'timestamp': pd.Timestamp(ts).strftime('%a %b %d %H:%M:%S +0000 %Y')
        })

# Output synthetic check-ins
df_synth = pd.DataFrame(synthetic_checkins)
df_synth.to_csv('3_synthetic_user_generation/synthetic_user_checkins.csv', index=False)
print(f"Generated {len(df_synth)} synthetic check-ins for {n_synth_users} synthetic users. Output saved to synthetic_user_checkins.csv.")
