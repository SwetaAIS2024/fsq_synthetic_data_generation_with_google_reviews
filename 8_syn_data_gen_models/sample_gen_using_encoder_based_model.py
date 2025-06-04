# Autoencoder-based Synthetic Data Generation for FSQ
# This script demonstrates how to use an autoencoder to generate synthetic user-POI check-in data.
# (PyTorch example, can be adapted to Keras/TensorFlow)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
import csv

# 1. Load FSQ check-in data
checkins = pd.read_csv('fsq_original_dataset/singapore_checkins.txt', sep='\t', names=['user_id', 'poi_id', 'timestamp', 'misc'])

# 2. Parse timestamps and assign temporal bins
checkins['parsed_time'] = pd.to_datetime(checkins['timestamp'], errors='coerce')

def get_time_of_day(dt):
    if pd.isnull(dt): return 'unknown'
    h = dt.hour
    if 5 <= h < 12: return 'morning'
    elif 12 <= h < 17: return 'afternoon'
    elif 17 <= h < 21: return 'evening'
    else: return 'night'

def get_week_bin(dt):
    if pd.isnull(dt): return 'unknown'
    return 'weekday' if dt.weekday() < 5 else 'weekend'

def get_season(dt):
    if pd.isnull(dt): return 'unknown'
    m = dt.month
    if m in [12, 1, 2]: return 'winter'
    elif m in [3, 4, 5]: return 'spring'
    elif m in [6, 7, 8]: return 'summer'
    else: return 'autumn'

checkins['time_of_day'] = checkins['parsed_time'].apply(get_time_of_day)
checkins['week_bin'] = checkins['parsed_time'].apply(get_week_bin)
checkins['season'] = checkins['parsed_time'].apply(get_season)

# Composite timebin label
checkins['timebin'] = checkins['time_of_day'] + '_' + checkins['week_bin'] + '_' + checkins['season']

# 3. Build user-POI-timebin sparse matrix (COO format)
user_ids = checkins['user_id'].unique()
poi_ids = checkins['poi_id'].unique()
timebins = checkins['timebin'].unique()
user_idx = {u: i for i, u in enumerate(user_ids)}
poi_idx = {p: i for i, p in enumerate(poi_ids)}
timebin_idx = {t: i for i, t in enumerate(timebins)}

user_indices = []
poi_indices = []
timebin_indices = []
data = []
for _, row in checkins.iterrows():
    u = user_idx[row['user_id']]
    p = poi_idx[row['poi_id']]
    t = timebin_idx[row['timebin']]
    user_indices.append(u)
    poi_indices.append(p)
    timebin_indices.append(t)
    data.append(1.0)

# Instead of a huge dense matrix, use a sparse representation
# Each entry is (user, poi, timebin) -> 1
# For autoencoder, you can treat each (poi, timebin) as a feature, but only for nonzero entries
# Here, we flatten (poi, timebin) into a single feature index
feature_indices = [p * len(timebins) + t for p, t in zip(poi_indices, timebin_indices)]
sparse_matrix = coo_matrix((data, (user_indices, feature_indices)), shape=(len(user_ids), len(poi_ids) * len(timebins)), dtype=np.float32)

# For deep learning, you can convert batches of this sparse matrix to dense as needed, or use only the nonzero entries for training
# Example: get a dense batch for a subset of users
# batch_users = np.random.choice(len(user_ids), size=128, replace=False)
# batch_X = sparse_matrix.tocsr()[batch_users].toarray()

print(f"Sparse matrix shape: {sparse_matrix.shape}, nnz: {sparse_matrix.nnz}")

# 4. Define a simple autoencoder
class AE(nn.Module):
    def __init__(self, n_in, n_latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, n_latent), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, 128), nn.ReLU(),
            nn.Linear(128, n_in), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# 5. Train the autoencoder using mini-batches to avoid memory issues
batch_size = 128
n_users = sparse_matrix.shape[0]
n_features = sparse_matrix.shape[1]
model = AE(n_in=n_features, n_latent=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(10):
    perm = np.random.permutation(n_users)
    for i in range(0, n_users, batch_size):
        batch_users = perm[i:i+batch_size]
        batch_X = sparse_matrix.tocsr()[batch_users].toarray()
        batch_X = (batch_X > 0).astype(np.float32)  # Ensure strictly 0 or 1
        batch_X = torch.tensor(batch_X, dtype=torch.float32)
        optimizer.zero_grad()
        out = model(batch_X)
        loss = loss_fn(out, batch_X)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. Generate synthetic users by sampling latent space, streaming to avoid OOM
import csv

n_synth = 100  # Reduced from 1000 to 100 to limit file size
latent_dim = 32
batch_synth = 10  # Number of synthetic users to process at once
output_path = '8_syn_data_gen_models/synthetic_checkins_autoencoder.csv'

# Precompute candidate timestamps for each (poi_id, timebin)
candidate_timestamps = {}
for (poi_id, timebin), group in checkins.groupby(['poi_id', 'timebin']):
    times = group['parsed_time'].dropna().values
    if len(times) > 0:
        candidate_timestamps[(poi_id, timebin)] = times
all_times = checkins['parsed_time'].dropna().values

# Write header
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['user_id', 'poi_id', 'timestamp', 'timebin'])
    writer.writeheader()

    synth_user_counter = 0
    for batch_start in range(0, n_synth, batch_synth):
        curr_batch = min(batch_synth, n_synth - batch_start)
        with torch.no_grad():
            latent_samples = torch.randn(curr_batch, latent_dim)
            synth_matrix = model.decoder(latent_samples).numpy()
            synth_matrix = (synth_matrix > 0.9).astype(int)  # Binarize with higher threshold for sparsity
        user_idx_arr, feat_idx_arr = np.nonzero(synth_matrix)
        for u, f in zip(user_idx_arr, feat_idx_arr):
            poi_idx_val = f // len(timebins)
            timebin_idx_val = f % len(timebins)
            poi_id = poi_ids[poi_idx_val]
            timebin = timebins[timebin_idx_val]
            candidates = candidate_timestamps.get((poi_id, timebin), None)
            if candidates is not None and len(candidates) > 0:
                ts = np.random.choice(candidates)
            else:
                ts = np.random.choice(all_times)
            ts_str = pd.Timestamp(ts).strftime('%a %b %d %H:%M:%S +0000 %Y')
            writer.writerow({
                'user_id': f'synth_user_{batch_start + u + 1}',
                'poi_id': poi_id,
                'timestamp': ts_str,
                'timebin': timebin
            })
        synth_user_counter += curr_batch
        print(f"Generated {synth_user_counter} synthetic users...")

print(f"Generated {n_synth} synthetic users using autoencoder. Output saved to synthetic_checkins_autoencoder.csv.")
