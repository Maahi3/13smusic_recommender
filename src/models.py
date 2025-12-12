import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
import os
from implicit.als import AlternatingLeastSquares
import faiss  # FAISS

print("=== TRAINING MODELS (YouTube metadata) ===")
print("Current directory:", os.getcwd())

# Load data
print("Loading data...")
features_df = pd.read_csv('data/processed/youtube_features.csv')
interactions = pd.read_pickle('data/processed/user_item_matrix.pkl')
print(f"Loaded {len(features_df)} videos, {len(interactions)} interactions")

# 1. POPULARITY (use YouTube view/like/comment normalized scores)
print("Training Popularity...")
features_df['popularity_score'] = (
    features_df.get('viewCount_norm', 0) * 0.7 +
    features_df.get('likeCount_norm', 0) * 0.2 +
    features_df.get('commentCount_norm', 0) * 0.1
)
popularity_scores = features_df[['video_id', 'popularity_score']].sort_values('popularity_score', ascending=False)
joblib.dump(popularity_scores, 'models/popularity.pkl')
print("Popularity saved.")

# 2. CONTENT-BASED WITH FAISS (text only)
print("Training Content-Based with FAISS...")
text_series = features_df.get('text')
if text_series is None or text_series.isna().all():
    raise RuntimeError("Missing 'text' column; run src/data_loader.py first.")

tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(text_series.fillna(''))
content_matrix = tfidf_matrix.astype('float32').toarray()

# Build FAISS index
d = content_matrix.shape[1]
faiss.normalize_L2(content_matrix)
index = faiss.IndexFlatIP(d)  # Inner Product = Cosine after L2 norm
index.add(content_matrix)

# Save FAISS index + ids
joblib.dump((index, features_df['video_id'].values), 'models/content.pkl')
print("FAISS Content model saved.")

# 3. ALS (on synthetic interactions)
print("Training ALS...")
user_item = coo_matrix((
    interactions['rating'],
    (interactions['user_id'].astype('category').cat.codes,
     interactions['track_id'].astype('category').cat.codes)
))
als = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=50)
als.fit(user_item.T)

track_codes = dict(enumerate(interactions['track_id'].astype('category').cat.categories))
user_codes = dict(enumerate(interactions['user_id'].astype('category').cat.categories))
joblib.dump((als, track_codes, user_codes), 'models/collab_als.pkl')
print("ALS saved.")

# 4. HYBRID DATA
hybrid_data = {
    'popularity_scores': popularity_scores,
    'content_index': index,
    'content_ids': features_df['video_id'].values,
    'interactions': interactions,
    'features_df': features_df,
    'track_codes': track_codes,
    'user_codes': user_codes,
    'tfidf_vocabulary': tfidf.vocabulary_,
}
joblib.dump(hybrid_data, 'models/hybrid_data.pkl')
print("Hybrid data saved.")

print("\nALL MODELS SAVED SUCCESSFULLY!")
print("Next: Run `streamlit run src/app.py`")