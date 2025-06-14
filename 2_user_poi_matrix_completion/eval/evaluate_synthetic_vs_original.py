# Evaluate Synthetic Check-ins vs. Original FSQ Check-ins (Matrix Completion)
# This script compares the synthetic check-ins generated by matrix completion
# to the original FSQ check-in dataset, using various statistical and distributional metrics.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
orig_checkins_path = 'fsq_original_dataset/singapore_checkins.txt'
synth_checkins_path = '2_user_poi_matrix_completion/synthetic_checkins_with_time.csv'

# Load original check-ins (tab-separated, no header)
df_orig = pd.read_csv(orig_checkins_path, sep='\t', names=['user_id', 'poi_id', 'timestamp', 'country_code'])

# Load synthetic check-ins (tab-separated, no header)
df_synth = pd.read_csv(synth_checkins_path, sep='\t', names=['user_id', 'poi_id', 'timestamp', 'country_code'])

# 1. Basic statistics
print('Original check-ins:', len(df_orig))
print('Synthetic check-ins:', len(df_synth))
print('Unique POIs (original):', df_orig['poi_id'].nunique())
print('Unique POIs (synthetic):', df_synth['poi_id'].nunique())
print('Unique users (original):', df_orig['user_id'].nunique())
print('Unique users (synthetic):', df_synth['user_id'].nunique())

# 2. Distribution: check-ins per POI
orig_poi_counts = df_orig['poi_id'].value_counts()
synth_poi_counts = df_synth['poi_id'].value_counts()

plt.figure(figsize=(10,5))
sns.histplot(orig_poi_counts, bins=50, color='blue', label='Original', stat='density', kde=True, alpha=0.5)
sns.histplot(synth_poi_counts, bins=50, color='orange', label='Synthetic', stat='density', kde=True, alpha=0.5)
plt.xlim(0, 300)
plt.legend()
plt.title('Distribution of Check-ins per POI')
plt.xlabel('Check-ins per POI')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('2_user_poi_matrix_completion/eval/checkins_per_poi_distribution.png')
plt.close()

# 3. Distribution: check-ins per user
orig_user_counts = df_orig['user_id'].value_counts()
synth_user_counts = df_synth['user_id'].value_counts()

plt.figure(figsize=(10,5))
sns.histplot(orig_user_counts, bins=50, color='blue', label='Original', stat='density', kde=True, alpha=0.5)
sns.histplot(synth_user_counts, bins=50, color='orange', label='Synthetic', stat='density', kde=True, alpha=0.5)
plt.xlim(0, 200)
plt.legend()
plt.title('Distribution of Check-ins per User')
plt.xlabel('Check-ins per User')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('2_user_poi_matrix_completion/eval/checkins_per_user_distribution.png')
plt.close()

# 4. Monthly distribution (if possible)
def extract_month(ts):
    try:
        return pd.to_datetime(ts, errors='coerce').strftime('%Y-%m')
    except Exception:
        return 'unknown'

df_orig['month'] = df_orig['timestamp'].apply(extract_month)
df_synth['month'] = df_synth['timestamp'].apply(extract_month)

orig_months = df_orig['month'].value_counts().sort_index()
synth_months = df_synth['month'].value_counts().sort_index()

plt.figure(figsize=(12,6))
plt.plot(orig_months.index, orig_months.values, label='Original', marker='o')
plt.plot(synth_months.index, synth_months.values, label='Synthetic', marker='x')
plt.title('Monthly Check-in Distribution: Real vs Synthetic')
plt.xlabel('Month')
plt.ylabel('Number of Check-ins')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('2_user_poi_matrix_completion/eval/monthly_checkin_counts.png')
plt.close()

# 5. Overlap analysis
common_pois = set(df_orig['poi_id']).intersection(set(df_synth['poi_id']))
print(f'POIs in both datasets: {len(common_pois)}')

# 6. Save summary stats
df_stats = pd.DataFrame({
    'dataset': ['original', 'synthetic'],
    'n_checkins': [len(df_orig), len(df_synth)],
    'n_unique_pois': [df_orig["poi_id"].nunique(), df_synth["poi_id"].nunique()],
    'n_unique_users': [df_orig["user_id"].nunique(), df_synth["user_id"].nunique()]
})
df_stats.to_csv('2_user_poi_matrix_completion/eval/synthetic_vs_original_stats.csv', index=False)

with open('2_user_poi_matrix_completion/eval/synthetic_vs_original_stats.txt', 'w') as f:
    f.write('Synthetic vs Original Check-in Dataset Statistics\n')
    f.write('==============================================\n')
    for i, row in df_stats.iterrows():
        f.write(f"Dataset: {row['dataset']}\n")
        f.write(f"  Number of check-ins: {row['n_checkins']}\n")
        f.write(f"  Number of unique POIs: {row['n_unique_pois']}\n")
        f.write(f"  Number of unique users: {row['n_unique_users']}\n")
        f.write('\n')
    f.write(f"POIs in both datasets: {len(common_pois)}\n")
    f.write(f"Average check-ins per user (original): {orig_user_counts.mean()}\n")
    f.write(f"Average check-ins per user (synthetic): {synth_user_counts.mean()}\n")
    f.write(f"Max check-ins by a user (original): {orig_user_counts.max()}\n")
    f.write(f"Users with >1000 check-ins (original): {(orig_user_counts > 1000).sum()}\n")
    f.write(f"Users with >1000 check-ins (synthetic): {(synth_user_counts > 1000).sum()}\n")

print('Evaluation complete. Plots and stats saved in eval/.')
