# Bootstrapping with Similar POIs
# This script generates synthetic check-ins for sparse POIs using patterns from similar POIs.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Load check-ins for FSQ (fix column order)
df = pd.read_csv('fsq_original_dataset/singapore_checkins.txt', sep='\t', names=['user_id', 'poi_id', 'timestamp', 'misc'])

# Count check-ins per POI
df_poi_counts = df['poi_id'].value_counts()

# Define sparse POIs (e.g., <= 5 check-ins)
sparse_pois = df_poi_counts[df_poi_counts <= 5].index.tolist()

# Identify non-sparse POIs (from both datasets)
non_sparse_pois = df_poi_counts[df_poi_counts > 5].index.tolist()
non_sparse_meta = all_poi_meta[all_poi_meta['poi_id'].isin(non_sparse_pois)]

# Use TF-IDF + cosine similarity for POI name matching
sparse_meta = all_poi_meta[all_poi_meta['poi_id'].isin(sparse_pois)]

# Drop rows with missing or blank POI names
def clean_poi_meta(meta):
    meta = meta.dropna(subset=['poi_name'])
    meta = meta[meta['poi_name'].astype(str).str.strip() != '']
    return meta

sparse_meta = clean_poi_meta(sparse_meta)
non_sparse_meta = clean_poi_meta(non_sparse_meta)

# Debugging: Print info about POI sets and metadata
print(f"Total unique POI IDs in check-ins: {df['poi_id'].nunique()}")
print(f"Total unique POI IDs in metadata: {all_poi_meta['poi_id'].nunique()}")
print(f"Sample POI IDs in check-ins: {df['poi_id'].astype(str).head(10).tolist()}")
print(f"Sample POI IDs in metadata: {all_poi_meta['poi_id'].astype(str).head(10).tolist()}")
print(f"POI ID dtype in check-ins: {df['poi_id'].dtype}")
print(f"POI ID dtype in metadata: {all_poi_meta['poi_id'].dtype}")
print(f"Sparse POIs (after cleaning): {len(sparse_meta)}")
print(f"Non-sparse POIs (after cleaning): {len(non_sparse_meta)}")
print("Sample sparse_meta:")
print(sparse_meta.head())
print("Sample non_sparse_meta:")
print(non_sparse_meta.head())

if sparse_meta.empty or non_sparse_meta.empty:
    print("No valid POI names for matching. Exiting.")
    exit(1)

# Prepare TF-IDF matrix for all POI names
vectorizer = TfidfVectorizer().fit(pd.concat([sparse_meta['poi_name'], non_sparse_meta['poi_name']]).astype(str))
sparse_name_vecs = vectorizer.transform(sparse_meta['poi_name'].astype(str))
non_sparse_name_vecs = vectorizer.transform(non_sparse_meta['poi_name'].astype(str))

synthetic_checkins = []
for i, (_, sparse_row) in enumerate(sparse_meta.iterrows()):
    sparse_poi = sparse_row['poi_id']
    sparse_name_vec = sparse_name_vecs[i]
    # Compute cosine similarity to all non-sparse POI names
    sims = cosine_similarity(sparse_name_vec, non_sparse_name_vecs)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    best_donor = non_sparse_meta.iloc[best_idx]['poi_id'] if best_score > 0.6 else None
    if best_donor:
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
