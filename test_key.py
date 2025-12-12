# test_key.py
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("YOUTUBE_API_KEY")

if not key:
    print("ERROR: Add YOUTUBE_API_KEY to .env")
else:
    print("Testing YouTube API key...")
    try:
        youtube = build("youtube", "v3", developerKey=key)
        res = youtube.search().list(part="snippet", q="Arijit Singh", maxResults=1).execute()
        print("KEY WORKS!")
        print("First song:", res["items"][0]["snippet"]["title"])
    except Exception as e:
        print("FAILED:", str(e))