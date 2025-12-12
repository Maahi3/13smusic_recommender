# Music Recommender

## Overview
A ML-based music recommendation system using Spotify API data. Supports content-based, collaborative filtering (SVD, ALS), popularity-based, and hybrid models. Includes EDA, modeling, testing, and Streamlit UI.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` with Spotify keys:
   ```
   SPOTIFY_CLIENT_ID=your_id
   SPOTIFY_CLIENT_SECRET=your_secret
   REDIRECT_URI=http://localhost:8888/callback
   ```
3. Fetch and process data: `python src/data_loader.py`
4. Train models: `python src/models.py`
5. EDA: Open `notebooks/01_eda.ipynb` in Jupyter.
6. Modeling/Eval: Open `notebooks/02_modeling.ipynb` in Jupyter.
7. Run app: `streamlit run src/app.py`
8. Tests: `pytest tests/`

## Workflow
- Data: Fetch from Spotify API, preprocess (clean, normalize), generate synthetic users.
- Models: Train and save .pkl files in models/.
- App: User/artist input for recs.
- Outputs: Metrics, plots, recommendations.csv.

## Enhancements
- Add real user data.
- Integrate FAISS for similarity.
- More APIs (e.g., Genius for lyrics).