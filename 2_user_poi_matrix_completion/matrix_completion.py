# User-POI Matrix Completion
# This script applies matrix completion techniques to the FSQ user-POI matrix to generate synthetic check-ins.

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import datetime

# 1. Load original FSQ check-ins
# Use the correct path relative to the project root
# dataset_path = '../fsq_original_dataset/singapore_checkins.txt'
dataset_path = 'fsq_original_dataset/singapore_checkins.txt'
# Assuming the file has columns: user_id, poi_id, timestamp, country_code also (tab-separated)
checkins = pd.read_csv(dataset_path, sep='\t', names=['user_id', 'poi_id', 'timestamp', 'country_code'], header=None) # co-pilot missed to add the country_code, this caused to fail the script
print(f"head of checkins: {checkins.head()}")

# Lower thresholds to allow more data through filtering
min_user_checkins = 1 #2
min_poi_checkins = 1 #2
user_counts = checkins['user_id'].value_counts()
poi_counts = checkins['poi_id'].value_counts()
active_users = user_counts[user_counts >= min_user_checkins].index
popular_pois = poi_counts[poi_counts >= min_poi_checkins].index
filtered_checkins = checkins[checkins['user_id'].isin(active_users) & checkins['poi_id'].isin(popular_pois)]

if filtered_checkins.empty:
    print("No data left after filtering. Try lowering min_user_checkins or min_poi_checkins.")
    exit(1)

# 2. Build user-POI interaction matrix (sparse)
user_idx = {u: i for i, u in enumerate(filtered_checkins['user_id'].unique())}
poi_idx = {p: i for i, p in enumerate(filtered_checkins['poi_id'].unique())}
rows = filtered_checkins['user_id'].map(user_idx)
cols = filtered_checkins['poi_id'].map(poi_idx)
data = np.ones(len(filtered_checkins))
sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_idx), len(poi_idx)))

# 3. Matrix completion using SVD (collaborative filtering)
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(sparse_matrix)
poi_factors = svd.components_.T  # shape: (num_pois, n_components)

# 4. Efficiently generate synthetic check-ins for zero entries with high predicted value
threshold = 0.2  # You may tune this threshold
max_synth_per_user = 100  # Limit to top-N synthetic check-ins per user
synthetic_checkins = []

# Build a mapping from poi_id to its real check-in timestamps
poi_timestamps = filtered_checkins.groupby('poi_id')['timestamp'].apply(list).to_dict()

for i, user in enumerate(user_idx):
    # user is the original user_id (should be int or str, as in FSQ)
    # Find POIs not visited by this user (i.e., zero entries)
    user_row = sparse_matrix.getrow(i).toarray().ravel()
    zero_poi_indices = np.where(user_row == 0)[0]
    pred_scores = np.dot(poi_factors[zero_poi_indices], user_factors[i])
    top_indices = np.argsort(-pred_scores)[:max_synth_per_user]
    for idx in top_indices:
        score = pred_scores[idx]
        if score > threshold:
            poi_j = zero_poi_indices[idx]
            poi = list(poi_idx.keys())[poi_j]
            # Sample a timestamp from real check-ins for this POI, or use a placeholder if none
            timestamps = poi_timestamps.get(poi, None)
            if timestamps:
                timestamp = np.random.choice(timestamps)
            else:
                timestamp = 'synthetic_timestamp'
            # Ensure user_id is always the original (int or str)
            synthetic_checkins.append({'user_id': user, 'poi_id': poi, 'timestamp': timestamp})

# 5. Output synthetic check-ins in correct FSQ format (tab-separated, no header)
synthetic_df = pd.DataFrame(synthetic_checkins)
synthetic_df['country_code'] = 480
synthetic_df = synthetic_df[['user_id', 'poi_id', 'timestamp', 'country_code']]
synthetic_df.to_csv('2_user_poi_matrix_completion/synthetic_checkins_with_time.csv', sep='\t', index=False, header=False)
print(f"Generated {len(synthetic_df)} synthetic check-ins with timestamps. Output saved to synthetic_checkins_with_time.csv.")

def extract_time_bin(ts, bin_type='month'):
    try:
        dt = pd.to_datetime(ts, errors='coerce')
        if pd.isnull(dt):
            return 'unknown'
        if bin_type == 'month':
            return dt.strftime('%Y-%m')
        elif bin_type == 'weekday':
            return dt.strftime('%A')
        elif bin_type == 'hour':
            return dt.strftime('%H')
        else:
            return dt.strftime('%Y-%m')
    except Exception:
        return 'unknown'

# Add a time bin column to filtered_checkins using robust parsing
filtered_checkins['time_bin'] = pd.to_datetime(filtered_checkins['timestamp'], errors='coerce').dt.strftime('%Y-%m')

# For each time bin, run SVD and generate synthetic check-ins
all_synth = []
time_bins = [tb for tb in filtered_checkins['time_bin'].unique() if tb not in [None, 'NaT', 'nan', 'unknown']]
for tbin in time_bins:
    bin_checkins = filtered_checkins[filtered_checkins['time_bin'] == tbin]
    if bin_checkins.empty:
        continue
    user_idx = {u: i for i, u in enumerate(bin_checkins['user_id'].unique())}
    poi_idx = {p: i for i, p in enumerate(bin_checkins['poi_id'].unique())}
    rows = bin_checkins['user_id'].map(user_idx)
    cols = bin_checkins['poi_id'].map(poi_idx)
    data = np.ones(len(bin_checkins))
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_idx), len(poi_idx)))
    if sparse_matrix.shape[0] < 2 or sparse_matrix.shape[1] < 2:
        continue
    svd = TruncatedSVD(n_components=min(20, min(sparse_matrix.shape)-1), random_state=42)
    user_factors = svd.fit_transform(sparse_matrix)
    poi_factors = svd.components_.T
    threshold = 0.2
    max_synth_per_user = 20
    for i, user in enumerate(user_idx):
        user_row = sparse_matrix.getrow(i).toarray().ravel()
        zero_poi_indices = np.where(user_row == 0)[0]
        pred_scores = np.dot(poi_factors[zero_poi_indices], user_factors[i])
        top_indices = np.argpartition(-pred_scores, max_synth_per_user-1)[:max_synth_per_user]
        for idx in top_indices:
            score = pred_scores[idx]
            if score > threshold:
                poi_j = zero_poi_indices[idx]
                poi = list(poi_idx.keys())[poi_j]
                # Use the first day of the month as a synthetic timestamp for the time bin
                try:
                    timestamp = pd.to_datetime(f'{tbin}-01')
                    timestamp = timestamp.strftime('%a %b %d 00:00:00 +0000 %Y')
                except Exception:
                    timestamp = 'synthetic_timestamp'
                all_synth.append({'user_id': user, 'poi_id': poi, 'timestamp': timestamp})

# Output synthetic check-ins with time bins in correct FSQ format (tab-separated, no header)
synthetic_df = pd.DataFrame(all_synth)
synthetic_df['country_code'] = 480
synthetic_df = synthetic_df[['user_id', 'poi_id', 'timestamp', 'country_code']]
synthetic_df.to_csv('2_user_poi_matrix_completion/synthetic_checkins_time_binned.csv', sep='\t', index=False, header=False)
print(f"Generated {len(synthetic_df)} time-binned synthetic check-ins. Output saved to synthetic_checkins_time_binned.csv.")
