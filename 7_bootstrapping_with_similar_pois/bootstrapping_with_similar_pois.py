# Bootstrapping with Similar POIs
# This script generates synthetic check-ins for sparse POIs using patterns from similar POIs.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import os

# Load POI metadata for both datasets
fsq_poi_meta = pd.read_csv('fsq_original_dataset/sg_poi_id_name.txt', sep='\t', names=['poi_id', 'poi_name'])
# For Google reviews, use place_id as poi_id and name as poi_name
google_poi_meta = pd.read_csv('google_reviews_in_fsq_format/all_reviews_with_timestamp.csv')
# Only keep columns if they exist
if 'place_id' in google_poi_meta.columns and 'name' in google_poi_meta.columns:
    google_poi_meta = google_poi_meta.rename(columns={'place_id': 'poi_id', 'name': 'poi_name'})
    google_poi_meta = google_poi_meta[['poi_id', 'poi_name']]
else:
    google_poi_meta['poi_name'] = 'unknown'
    google_poi_meta = google_poi_meta[['poi_id', 'poi_name']]

# Combine POI metadata for similarity search
all_poi_meta = pd.concat([fsq_poi_meta, google_poi_meta]).drop_duplicates('poi_id')

# Load check-ins for FSQ
df = pd.read_csv('fsq_original_dataset/singapore_checkins.txt', sep='\t', names=['user_id', 'poi_id', 'timestamp'])

# Count check-ins per POI
df_poi_counts = df['poi_id'].value_counts()

# Define sparse POIs (e.g., <= 5 check-ins)
sparse_pois = df_poi_counts[df_poi_counts <= 5].index.tolist()

# Use POI name string similarity as a proxy for similarity
from difflib import SequenceMatcher

def approx_match(a, b, threshold=0.6):
    return SequenceMatcher(None, a, b).ratio() > threshold

# Identify non-sparse POIs (from both datasets)
non_sparse_pois = df_poi_counts[df_poi_counts > 5].index.tolist()
non_sparse_meta = all_poi_meta[all_poi_meta['poi_id'].isin(non_sparse_pois)]

synthetic_checkins = []
for sparse_poi in sparse_pois:
    sparse_row = all_poi_meta[all_poi_meta['poi_id'] == sparse_poi]
    if sparse_row.empty:
        continue
    sparse_name = sparse_row.iloc[0]['poi_name']
    # Find the most similar non-sparse POI by approximate string match
    best_score = 0
    best_donor = None
    for _, donor_row in non_sparse_meta.iterrows():
        donor_name = donor_row['poi_name']
        score = SequenceMatcher(None, str(sparse_name), str(donor_name)).ratio()
        if score > best_score:
            best_score = score
            best_donor = donor_row['poi_id']
    if best_donor and best_score > 0.6:
        # Try to get donor check-ins from FSQ, else from Google reviews
        donor_checkins = df[df['poi_id'] == best_donor]
        if donor_checkins.empty:
            # Try from Google reviews
            google_reviews = pd.read_csv('google_reviews_in_fsq_format/all_reviews_with_timestamp.csv')
            if 'place_id' in google_reviews.columns:
                donor_checkins = google_reviews[google_reviews['place_id'] == best_donor]
            else:
                donor_checkins = google_reviews[google_reviews['poi_id'] == best_donor]
        for _, row in donor_checkins.iterrows():
            synthetic_checkins.append({
                'user_id': row['user_id'] if 'user_id' in row else 'synthetic_user',
                'poi_id': sparse_poi,
                'timestamp': row['timestamp'],
                'source_poi': best_donor,
                'match_score': best_score
            })

synthetic_df = pd.DataFrame(synthetic_checkins)
synthetic_df.to_csv('synthetic_checkins_bootstrapped.csv', index=False)
print(f"Generated {len(synthetic_df)} synthetic check-ins for sparse POIs. Output saved to synthetic_checkins_bootstrapped.csv.")
