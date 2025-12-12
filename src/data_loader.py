# src/data_loader.py
import os
import pandas as pd
import numpy as np
import pickle
from googleapiclient.discovery import build
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

ARTISTS = [
    "Arijit Singh", "Ed Sheeran", "Taylor Swift", "The Weeknd", "Dua Lipa",
    "Pritam", "Badshah", "Diljit Dosanjh", "AP Dhillon", "Shreya Ghoshal"
]

def fetch_youtube_videos():
    os.makedirs("data/processed", exist_ok=True)
    videos = []

    for artist in ARTISTS:
        print(f"Fetching videos for {artist}...")
        try:
            res = youtube.search().list(
                part="snippet",
                q=f"{artist} official music video",
                type="video",
                maxResults=50,
                order="viewCount"
            ).execute()
            for item in res.get("items", []):
                stats = youtube.videos().list(
                    part="statistics",
                    id=item["id"]["videoId"]
                ).execute()
                views = int(stats["items"][0]["statistics"].get("viewCount", 0)) if stats["items"] else 0
                videos.append({
                    "video_id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "artist": artist,
                    "views": views,
                    "text": f"{item['snippet']['title']} {artist}"
                })
        except Exception as e:
            print(f"Error fetching {artist}: {e}")

    df = pd.DataFrame(videos).drop_duplicates(subset="video_id")
    df.to_csv("data/processed/youtube_videos.csv", index=False)
    print(f"Saved {len(df)} videos.")
    return df

def generate_features_and_interactions(df):
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])

    # Synthetic audio features
    np.random.seed(42)
    audio_features = pd.DataFrame({
        col: np.random.uniform(0, 1, len(df))
        for col in ['danceability', 'energy', 'valence', 'acousticness']
    })
    df = pd.concat([df, audio_features], axis=1)

    # Combine
    content_matrix = hstack([tfidf_matrix, audio_features.values]).toarray().astype('float32')

    # Synthetic user-item
    num_users = 1000
    interactions = []
    for u in range(num_users):
        user_id = f"user_{u}"
        chosen = np.random.choice(df['video_id'], size=np.random.randint(10, 30), replace=False)
        for vid in chosen:
            interactions.append({"user_id": user_id, "video_id": vid, "rating": np.random.randint(1, 6)})
    interaction_df = pd.DataFrame(interactions)
    pickle.dump(interaction_df, open("data/processed/user_item_matrix.pkl", "wb"))

    df.to_csv("data/processed/youtube_features.csv", index=False)
    joblib.dump(content_matrix, "data/processed/content_matrix.pkl")
    print("Features + interactions ready.")
    return df, content_matrix, interaction_df

if __name__ == "__main__":
    import joblib
    df = fetch_youtube_videos()
    df, matrix, interactions = generate_features_and_interactions(df)
    print("All data ready!")