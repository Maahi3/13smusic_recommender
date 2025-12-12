import pytest
import pandas as pd
import pickle
from src.models import ContentBased, PopularityBased

def precision_at_k(relevant, k):
    return sum(relevant[:k]) / k if k > 0 else 0

def test_data_matrix():
    with open('data/processed/user_item_matrix.pkl', 'rb') as f:
        df = pickle.load(f)
    assert len(df) > 0
    assert 'user_id' in df.columns
    assert 'track_id' in df.columns
    assert 'rating' in df.columns

def test_content_recommend():
    df = pd.read_csv('data/processed/spotify_features.csv')
    model = ContentBased(df)
    seed_track = df['track_id'].iloc[0]
    recs = model.recommend(seed_track, n=5)
    assert len(recs) == 5

def test_popularity_recommend():
    df = pd.read_csv('data/processed/spotify_features.csv')
    model = PopularityBased(df)
    recs = model.recommend(5)
    assert len(recs) == 5
    assert all(id in df['track_id'].values for id in recs)

def test_precision_at_k():
    relevant = [1, 0, 1, 0, 1]
    assert abs(precision_at_k(relevant, 3) - 2/3) < 1e-6