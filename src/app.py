import os
import json
import random
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(page_title="Music Recommender", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed", "youtube_features.csv")
HISTORY_DIR = os.path.join(DATA_DIR, "user_history")
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

ARTISTS = [
    "Arijit Singh", "Pritam", "Ed Sheeran", "Badshah", "Diljit Dosanjh",
    "AP Dhillon", "Drake", "Taylor Swift", "The Weeknd", "Dua Lipa",
    "Shreya Ghoshal", "Atif Aslam", "Neha Kakkar", "Jubin Nautiyal", "Armaan Malik",
    "B Praak", "Darshan Raval", "Honey Singh", "Guru Randhawa", "Billie Eilish",
    "KK", "A. R. Rahman", "Sonu Nigam", "Shankar Mahadevan", "Sunidhi Chauhan"
]

# ============================================================================
# DARK THEME CSS
# ============================================================================
st.markdown(
    """
    <style>
      div.block-container { padding-top: 0rem !important; padding-bottom: 0.25rem !important; }
      /* keep the Streamlit header so the sidebar toggle remains accessible */
      /* header[data-testid="stHeader"] { display: none !important; }  <-- removed */

      footer { display: none !important; }
      
      .song-card { background-color: #282828; padding: 8px; border-radius: 8px; margin: 6px 0; color: #fff; }
      .small-caption { color: #b3b3b3; font-size: 0.9rem; }
      .profile-container { display:flex; align-items:center; gap:12px; padding:12px; margin-top:20px; background:#1a1a1a; border-radius:8px; }
      .profile-avatar { width:50px; height:50px; border-radius:50%; background:#fff; display:flex; align-items:center; justify-content:center; font-size:24px; color:#000; font-weight:bold; }
      .compact-btn { background:#1DB954; color:#fff; border:none; padding:6px 10px; border-radius:6px; cursor:pointer; font-weight:500; font-size:0.9rem; }
      .compact-btn:hover { background:#1ed760; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# YOUTUBE PLAYER INJECTION (iframe hosts player; parent <-> iframe communicate via postMessage)
# ============================================================================
def inject_youtube_api_and_player():
    """
    Parent-side injection: create/update a YouTube embed iframe in the main page (off-screen)
    and expose window.playVideo / pausePlayer / stopPlayer / seekToSeconds.
    Creating/updating the iframe in the parent ensures the play call happens as part of
    the user's gesture (avoid autoplay blocking).
    """
    js = r'''
<script>
(function(){
  if (window.__st_yt_parent_injected) return;
  window.__st_yt_parent_injected = true;

  var containerId = 'yt_player_container_st';
  var iframeId = 'yt_player_iframe_st';
  var currentVideoId = null;
  window.__st_player_duration = 0;

  function ensureContainer(){
    if (document.getElementById(containerId)) return;
    var c = document.createElement('div');
    c.id = containerId;
    c.style.position = 'fixed';
    c.style.left = '-9999px';
    c.style.top = '0'; 
    c.style.width = '320px';
    c.style.height = '180px';
    c.style.overflow = 'hidden';
    c.style.zIndex = '999999';
    document.body.appendChild(c);
  }

  function createOrUpdateIframe(videoId, autoplay){
    ensureContainer();
    var src = 'https://www.youtube.com/embed/' + encodeURIComponent(videoId) + '?enablejsapi=1&autoplay=' + (autoplay?1:0) + '&rel=0&modestbranding=1&playsinline=1';
    var iframe = document.getElementById(iframeId);
    if (!iframe) {
      iframe = document.createElement('iframe');
      iframe.id = iframeId;
      iframe.width = '320';
      iframe.height = '180';
      iframe.frameBorder = '0';
      iframe.allow = 'autoplay; encrypted-media';
      iframe.allowFullscreen = true;
      iframe.src = src;
      document.getElementById(containerId).appendChild(iframe);
    } else {
      if (iframe.src !== src) iframe.src = src;
    }
    return iframe;
  }

  function postCommand(cmd, args){
    try {
      var iframe = document.getElementById(iframeId);
      if (!iframe || !iframe.contentWindow) return;
      var msg = JSON.stringify({ event: 'command', func: cmd, args: args || [] });
      iframe.contentWindow.postMessage(msg, '*');
    } catch(e){}
  }

  // Expose API to song card buttons (calls run in user gesture)
  window.playVideo = function(id, title, thumb){
    try {
      currentVideoId = id;
      // create or update iframe with autoplay=1 inside the user's click gesture
      createOrUpdateIframe(id, true);
      // best-effort: ask iframe to play via postMessage after a short delay
      setTimeout(function(){ postCommand('playVideo', []); }, 300);

      // populate global remote object so mini UI handlers (existing code) can access it
      window.__st_yt_remote = window.__st_yt_remote || {};
      window.__st_yt_remote.currentVideoId = id;
      window.__st_yt_remote.currentTitle = title || '';
      window.__st_yt_remote.currentThumb = thumb || '';

      // notify existing mini_ui_handler which listens for postMessage { type: 'yt:mini_show' }
      try {
        window.postMessage({ type: 'yt:mini_show', title: title || '', thumb: thumb || '' }, '*');
      } catch(e){}

      // also dispatch a CustomEvent for any listener expecting it
      try {
        var ev = new CustomEvent('st:yt_mini_show', { detail: { title: title || '', thumb: thumb || '' }});
        window.dispatchEvent(ev);
      } catch(e){}

      return true;
    } catch(e){
      try { alert('Playback failed to start'); } catch(_) {}
      return false;
    }
  };
  window.pausePlayer = function(){ try { postCommand('pauseVideo', []); } catch(e){} };
  window.stopPlayer = function(){ try { postCommand('stopVideo', []); var ev = new CustomEvent('st:yt_mini_hide'); window.dispatchEvent(ev); window.postMessage({ type: 'yt:mini_hide' }, '*'); } catch(e){} };
  window.seekToSeconds = function(s){ try { postCommand('seekTo', [s, true]); } catch(e){} };

  // Listen for messages from YT iframe (infoDelivery / state changes) and forward as simple events
  window.addEventListener('message', function(e){
    try {
      var data = e.data;
      if (typeof data === 'string') {
        try { data = JSON.parse(data); } catch(err){ return; }
      }
      if (!data) return;
      // YouTube will send infoDelivery events containing currentTime/duration etc.
      if (data.event === 'infoDelivery' && data.info) {
        var cur = data.info.currentTime || 0;
        var dur = data.info.duration || window.__st_player_duration || 0;
        var pct = (dur > 0) ? ((cur/dur)*100) : 0;
        // forward using CustomEvent so existing mini UI code can pick it up
        window.dispatchEvent(new CustomEvent('st:yt_progress', { detail: { cur: cur, dur: dur, pct: pct } }));
        // also forward as the message shape mini_ui_handler expects
        window.postMessage({ type: 'yt:progress', cur: cur, dur: dur, pct: pct }, '*');
      }
      if (data.event === 'onStateChange') {
        // forward state change as well
        window.dispatchEvent(new CustomEvent('st:yt_state', { detail: { state: data.info } }));
        window.postMessage({ type: 'yt:state', state: data.info }, '*');
      }
    } catch(e){}
  }, false);

})();
</script>
'''
    st.markdown(js, unsafe_allow_html=True)


# ============================================================================
# YOUTUBE & CATALOG UTILITIES
# ============================================================================
def get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY not found in environment. Add it to .env")
    return build("youtube", "v3", developerKey=api_key)


@st.cache_resource
def load_catalog():
    if os.path.exists(PROCESSED_CSV):
        df = pd.read_csv(PROCESSED_CSV)
        def col_or_empty(col):
            return df[col].fillna("").astype(str) if col in df.columns else pd.Series([""] * len(df), index=df.index)
        if "text" not in df.columns:
            df["text"] = (
                col_or_empty("title") + " " +
                col_or_empty("description") + " " +
                col_or_empty("tags") + " " +
                col_or_empty("channel") + " " +
                col_or_empty("artist")
            ).str[:10000]
        return df
    return pd.DataFrame(columns=["video_id", "title", "channel", "text", "viewCount_norm"])


@st.cache_resource
def build_tfidf_matrix(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None
    vec = TfidfVectorizer(max_features=10000, stop_words="english")
    X = vec.fit_transform(df["text"].fillna(""))
    return vec, X


# ============================================================================
# HISTORY UTILITIES
# ============================================================================
def _history_path(user_id: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    return os.path.join(HISTORY_DIR, f"{user_id}.json")


def load_history(user_id: str) -> List[str]:
    path = _history_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def save_history(user_id: str, history: List[str]) -> Tuple[bool, str, str]:
    path = _history_path(user_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history or [], f, ensure_ascii=False, indent=2)
        return True, path, None
    except Exception as e:
        return False, path, str(e)


def add_to_history(user_id: str, video_id: str):
    if not video_id:
        return False, None, "Empty video id"
    history = load_history(user_id)
    if video_id not in history:
        history.append(video_id)
        return save_history(user_id, history)
    return True, _history_path(user_id), None


def get_active_user_id() -> str:
    uid = st.session_state.get("user_id")
    if uid and isinstance(uid, str) and uid.strip():
        return uid.strip()
    return "me"


# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================
def get_exactly_10(artist_name=None):
    name = (artist_name or random.choice(ARTISTS)).strip()
    try:
        youtube = get_youtube_client()
    except Exception:
        youtube = None

    candidates = []
    tried = set()
    queries = [
        f"{name} official audio",
        f"{name} official music video",
        f"{name} music video",
        f"{name} song",
        f"{name} audio"
    ]

    if youtube:
        for q in queries:
            try:
                res = youtube.search().list(
                    part="snippet",
                    q=q,
                    maxResults=50,
                    type="video",
                    safeSearch="none"
                ).execute()
                for item in res.get("items", []):
                    vid = (item.get("id") or {}).get("videoId")
                    if vid and vid not in tried:
                        candidates.append(item)
                        tried.add(vid)
                if len(candidates) >= 10:
                    break
            except Exception:
                continue

    if len(candidates) < 10:
        df = load_catalog()
        if not df.empty:
            name_l = name.lower()
            mask = pd.Series(False, index=df.index)
            for col in ["artist", "channel", "title", "description", "tags"]:
                if col in df.columns:
                    mask = mask | df[col].astype(str).str.lower().str.contains(name_l, na=False)
            remaining = df[mask & ~df["video_id"].astype(str).isin(tried)]
            if not remaining.empty:
                if "viewCount_norm" in remaining.columns:
                    remaining = remaining.sort_values("viewCount_norm", ascending=False)
                needed = 10 - len(candidates)
                for _, r in remaining.head(needed).iterrows():
                    vid = str(r.get("video_id"))
                    candidates.append({
                        "id": {"videoId": vid},
                        "snippet": {"title": r.get("title", "") or "", "channelTitle": r.get("channel", "") or ""}
                    })
                    tried.add(vid)

    random.shuffle(candidates)
    return candidates[:10]


def get_any_10():
    results = []
    seen = set()
    artist_candidates = ARTISTS[:] if len(ARTISTS) <= 8 else random.sample(ARTISTS, 8)
    for a in artist_candidates:
        try:
            for it in get_exactly_10(a):
                vid = (it.get("id") or {}).get("videoId")
                if vid and vid not in seen:
                    results.append(it)
                    seen.add(vid)
                if len(results) >= 10:
                    break
        except Exception:
            continue
        if len(results) >= 10:
            break

    if not results:
        df = load_catalog()
        if not df.empty:
            if "viewCount_norm" in df.columns:
                samp = df.sort_values("viewCount_norm", ascending=False).head(10)
            else:
                samp = df.sample(min(10, len(df)), random_state=42)
            for _, r in samp.iterrows():
                vid = str(r.get("video_id"))
                results.append({"id": {"videoId": vid}, "snippet": {"title": r.get("title", ""), "channelTitle": r.get("channel", "")}})
    return results[:10]


def recommend_for_user(user_id: str, top_k: int = 10):
    df = load_catalog()
    if df is None or df.empty:
        return get_any_10()

    vec, X = build_tfidf_matrix(df)
    if X is None:
        return get_any_10()

    history = load_history(user_id) or []
    if not history:
        if "viewCount_norm" in df.columns and df["viewCount_norm"].notna().any():
            top = df.sort_values("viewCount_norm", ascending=False).head(top_k)
        else:
            top = df.sample(min(top_k, len(df)), random_state=42) if len(df) else df
        items = []
        for _, r in top.head(top_k).iterrows():
            vid = str(r.get("video_id"))
            items.append({"id": {"videoId": vid}, "snippet": {"title": r.get("title", ""), "channelTitle": r.get("channel", "")}})
        if len(items) < top_k and len(df) > len(items):
            remaining = df[~df["video_id"].astype(str).isin([it["id"]["videoId"] for it in items])]
            if len(remaining) > 0:
                extra = remaining.sample(min(top_k - len(items), len(remaining)), random_state=42)
                for _, r in extra.iterrows():
                    vid = str(r.get("video_id"))
                    items.append({"id": {"videoId": vid}, "snippet": {"title": r.get("title", ""), "channelTitle": r.get("channel", "")}})
        return items[:top_k]

    id_to_idx = {vid: i for i, vid in enumerate(df["video_id"].astype(str).tolist())}
    hist_idx = [id_to_idx[v] for v in history if v in id_to_idx]
    if not hist_idx:
        if "viewCount_norm" in df.columns:
            fallback = df.sort_values("viewCount_norm", ascending=False).head(top_k)
        else:
            fallback = df.sample(min(top_k, len(df)), random_state=42)
        items = []
        for _, r in fallback.iterrows():
            vid = str(r.get("video_id"))
            items.append({"id": {"videoId": vid}, "snippet": {"title": r.get("title", ""), "channelTitle": r.get("channel", "")}})
        if len(items) < top_k and len(df) > len(items):
            remaining = df[~df["video_id"].astype(str).isin([it["id"]["videoId"] for it in items])]
            if len(remaining) > 0:
                extra = remaining.sample(min(top_k - len(items), len(remaining)), random_state=42)
                for _, r in extra.iterrows():
                    vid = str(r.get("video_id"))
                    items.append({"id": {"videoId": vid}, "snippet": {"title": r.get("title", ""), "channelTitle": r.get("channel", "")}})
        return items[:top_k]

    sims = cosine_similarity(X[hist_idx], X).mean(axis=0)
    sims = np.array(sims).ravel()
    for i in hist_idx:
        sims[i] = -1.0
    top_idx = np.argsort(-sims)[:top_k]
    items = []
    for i in top_idx:
        r = df.iloc[i]
        vid = str(r.get("video_id"))
        items.append({"id": {"videoId": vid}, "snippet": {"title": r.get("title", ""), "channelTitle": r.get("channel", "")}})

    if len(items) < top_k and len(df) > len(items):
        existing_ids = set(it["id"]["videoId"] for it in items)
        remaining = df[~df["video_id"].astype(str).isin(existing_ids)]
        if "viewCount_norm" in remaining.columns:
            remaining = remaining.sort_values("viewCount_norm", ascending=False)
        if len(remaining) > 0:
            extra = remaining.head(top_k - len(items))
            for _, r in extra.iterrows():
                vid = str(r.get("video_id"))
                items.append({"id": {"videoId": vid}, "snippet": {"title": r.get("title", ""), "channelTitle": r.get("channel", "")}})
    return items[:top_k]


# ============================================================================
# CATALOG HELPERS
# ============================================================================
def ensure_in_catalog(video_id: str):
    if not video_id:
        return False, "empty video id"
    try:
        df = load_catalog()
    except Exception:
        df = pd.DataFrame()
    if not df.empty and "video_id" in df.columns and str(video_id) in df["video_id"].astype(str).tolist():
        return True, None
    try:
        yt = get_youtube_client()
        res = yt.videos().list(part="snippet,statistics", id=video_id).execute()
        items = res.get("items", [])
    except Exception as e:
        return False, f"YouTube API error: {e}"
    if not items:
        return False, "video not found via YouTube API"
    v = items[0]
    sn = v.get("snippet", {})
    stt = v.get("statistics", {})
    new_row = {
        "video_id": video_id,
        "title": sn.get("title", ""),
        "channel": sn.get("channelTitle", ""),
        "artist": "",
        "tags": ",".join(sn.get("tags", [])) if sn.get("tags") else "",
        "description": sn.get("description", ""),
        "text": " ".join(filter(None, [
            sn.get("title", ""),
            sn.get("description", ""),
            ",".join(sn.get("tags", [])) if sn.get("tags") else "",
            sn.get("channelTitle", "")
        ]))[:10000],
        "viewCount": int(stt.get("viewCount", 0)) if stt.get("viewCount") else 0,
        "likeCount": int(stt.get("likeCount", 0)) if stt.get("likeCount") else 0,
        "commentCount": int(stt.get("commentCount", 0)) if stt.get("commentCount") else 0,
        "viewCount_norm": 0.0,
        "likeCount_norm": 0.0,
        "commentCount_norm": 0.0,
    }
    try:
        if df is None or df.empty:
            df = pd.DataFrame([new_row])
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
        df.to_csv(PROCESSED_CSV, index=False)
    except Exception as e:
        return False, f"Failed to write catalog CSV: {e}"
    try:
        load_catalog.clear()
    except Exception:
        pass
    try:
        build_tfidf_matrix.clear()
    except Exception:
        pass
    return True, None


def save_to_library(video_id: str, title: str, user_id: str = None):
    if user_id is None:
        user_id = get_active_user_id()
    ok_cat, err_cat = ensure_in_catalog(video_id)
    ok, path, err = add_to_history(user_id, video_id)
    if ok:
        st.success(f"✅ '{title}' added to your library!")
        return True
    else:
        st.error(f"Failed to save '{title}': {err or err_cat}")
        return False


# ============================================================================
# CARD RENDERER (calls window.playVideo via postMessage wrapper)
# ============================================================================
def render_song_card(idx: int, vid: str, title: str, channel: str):
    """Render a compact song card with Play link that opens the video on YouTube (no in-app player)."""
    thumb = f"https://img.youtube.com/vi/{vid}/default.jpg"
    safe_title = (title or "").replace("'", "&#39;").replace('"', "&quot;")
    safe_channel = (channel or "").replace("'", "&#39;").replace('"', "&quot;")
    yt_url = f"https://www.youtube.com/watch?v={vid}"

    html = f"""
<div class="song-card">
  <div style="display:flex; align-items:center; gap:10px;">
    <img src="{thumb}" style="width:56px; height:56px; object-fit:cover; border-radius:6px;" />
    <div style="flex:1;">
      <div style="display:flex; justify-content:space-between; align-items:start;">
        <div>
          <strong style="font-size:0.95rem;">{idx}. {safe_title}</strong>
          <div class="small-caption">by {safe_channel}</div>
        </div>
        <a class="compact-btn" href="{yt_url}" target="_blank" rel="noopener noreferrer" title="Open on YouTube">▶️ Play</a>
      </div>
    </div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# STREAMLIT APP INITIALIZATION
# ============================================================================
if "yt_component_injected" not in st.session_state:
    st.session_state["yt_component_injected"] = True
    inject_youtube_api_and_player()

# Parent-side mini UI installer: listens for iframe messages and updates mini player in main DOM
mini_ui_handler = '''
<script>
(function(){
  if (window.__st_yt_mini_ui_installed) return;
  window.__st_yt_mini_ui_installed = true;

  function ensureMiniUI(){
    if (document.getElementById('st_mini_player_container')) return;
    var cont = document.createElement('div');
    cont.id = 'st_mini_player_container';
    cont.style.position = 'fixed';
    cont.style.bottom = '20px';
    cont.style.right = '20px';
    cont.style.width = '320px';
    cont.style.background = '#181818';
    cont.style.borderRadius = '8px';
    cont.style.padding = '12px';
    cont.style.color = '#fff';
    cont.style.zIndex = '9999';
    cont.style.boxShadow = '0 4px 12px rgba(0,0,0,0.5)';
    cont.style.display = 'none';
    cont.innerHTML = `
      <div style="display:flex; gap:10px; align-items:center;">
        <img id="st_mini_thumb" src="" style="width:72px;height:72px;object-fit:cover;border-radius:6px;background:#000;" />
        <div style="flex:1;display:flex;flex-direction:column;gap:6px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div id="st_mini_title" style="font-weight:700;font-size:0.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">No track</div>
            <div style="display:flex;gap:8px;">
              <button id="st_btn_play" style="background:#1DB954;color:#fff;border:none;padding:6px 10px;border-radius:6px;cursor:pointer;">▶️</button>
              <button id="st_btn_pause" style="background:#FFA500;color:#fff;border:none;padding:6px 10px;border-radius:6px;cursor:pointer;">⏸️</button>
              <button id="st_btn_close" style="background:#444;color:#fff;border:none;padding:6px 10px;border-radius:6px;cursor:pointer;">✕</button>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <div id="st_time_cur" style="color:#b3b3b3;font-size:0.85rem;min-width:34px;">0:00</div>
            <div id="st_progress" style="flex:1;height:6px;background:#2e2e2e;border-radius:4px;cursor:pointer;position:relative;">
              <div id="st_progress_fill" style="height:100%;background:#1DB954;width:0%;border-radius:4px;"></div>
            </div>
            <div id="st_time_dur" style="color:#b3b3b3;font-size:0.85rem;min-width:34px;">0:00</div>
          </div>
        </div>
      </div>`;
    document.body.appendChild(cont);

    setTimeout(function(){
      try {
        var play = document.getElementById('st_btn_play');
        var pause = document.getElementById('st_btn_pause');
        var close = document.getElementById('st_btn_close');
        var prog = document.getElementById('st_progress');

        if (play) play.addEventListener('click', function(){ try { if (window.__st_yt_remote && window.__st_yt_remote.currentVideoId) window.playVideo(window.__st_yt_remote.currentVideoId, window.__st_yt_remote.currentTitle, window.__st_yt_remote.currentThumb); } catch(e){} });
        if (pause) pause.addEventListener('click', function(){ try { window.pausePlayer(); } catch(e){} });
        if (close) close.addEventListener('click', function(){ try { window.stopPlayer(); } catch(e){} });

        if (prog) prog.addEventListener('click', function(ev){
          try {
            var rect = this.getBoundingClientRect();
            var pct = (ev.clientX - rect.left) / rect.width;
            var dur = window.__st_player_duration || 0;
            if (dur > 0) window.seekToSeconds(pct * dur);
          } catch(e){}
        });
      } catch(e){}
    }, 200);
  }

  ensureMiniUI();

  window.addEventListener('message', function(e){
    try {
      var d = e && e.data;
      if (!d || typeof d !== 'object') return;
      if (d.type === 'yt:mini_show') {
        ensureMiniUI();
        var cont = document.getElementById('st_mini_player_container');
        if (cont) cont.style.display = 'flex';
        var title = d.title || '';
        var thumb = d.thumb || '';
        var titleEl = document.getElementById('st_mini_title'); if (titleEl) titleEl.textContent = title || 'Unknown';
        var thumbEl = document.getElementById('st_mini_thumb'); if (thumbEl && thumb) thumbEl.src = thumb;
        window.__st_yt_remote = window.__st_yt_remote || {};
        window.__st_yt_remote.currentTitle = title || '';
        window.__st_yt_remote.currentThumb = thumb || '';
      } else if (d.type === 'yt:mini_hide') {
        var cont = document.getElementById('st_mini_player_container');
        if (cont) cont.style.display = 'none';
      } else if (d.type === 'yt:progress') {
        var fill = document.getElementById('st_progress_fill'); if (fill) fill.style.width = (d.pct || 0) + '%';
        var tc = document.getElementById('st_time_cur'); if (tc) tc.textContent = (function(s){ if (!s||isNaN(s)) return '0:00'; var m=Math.floor(s/60), sec=Math.floor(s%60); return m+':'+(sec<10?'0'+sec:sec); })(d.cur || 0);
        var td = document.getElementById('st_time_dur'); if (td) td.textContent = (function(s){ if (!s||isNaN(s)) return '0:00'; var m=Math.floor(s/60), sec=Math.floor(s%60); return m+':'+(sec<10?'0'+sec:sec); })(d.dur || 0);
        window.__st_player_duration = d.dur || 0;
      } else if (d.type === 'yt:ready_failed') {
        alert('YouTube player failed to initialize.');
      }
    } catch(err) { console.warn(err); }
  }, false);
})();
</script>
'''
st.markdown(mini_ui_handler, unsafe_allow_html=True)

st.title("🎵 Music Recommender")
st.caption("Discover music based on your taste or explore by artist")

# Session state defaults
if "user_id" not in st.session_state or not st.session_state.get("user_id"):
    st.session_state["user_id"] = "me"
if "user_recs" not in st.session_state:
    st.session_state["user_recs"] = []
if "artist_results" not in st.session_state:
    st.session_state["artist_results"] = []
if "artist" not in st.session_state:
    st.session_state["artist"] = ARTISTS[0]
if "artist_model" not in st.session_state:
    st.session_state["artist_model"] = "Hybrid"

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    mode = st.radio("Mode", ["User", "Artist"])
    
    if mode == "User":
        user_id = st.text_input("User ID", get_active_user_id()).strip() or "me"
        st.session_state["user_id"] = user_id
    else:
        artist = st.selectbox("Artist", ARTISTS, index=ARTISTS.index(st.session_state.get("artist", ARTISTS[0])))
        model = st.selectbox("Model", ["Hybrid", "Popularity", "Content-Based"], index=["Hybrid", "Popularity", "Content-Based"].index(st.session_state.get("artist_model", "Hybrid")))
        st.session_state["artist"] = artist
        st.session_state["artist_model"] = model

    st.markdown("---")
    st.markdown(
        """
        <div class="profile-container">
            <div class="profile-avatar">🎧</div>
            <div style="display:flex; flex-direction:column;">
                <div style="font-weight:700; color:#fff; font-size:0.95rem;">maahi</div>
                <div style="color:#b3b3b3; font-size:0.85rem;">Active user</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# USER MODE
# ============================================================================
if mode == "User":
    st.subheader("Your Recommendations")
    
    def _safe_rerun():
        try:
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                # fallback: stop execution (Streamlit will preserve updated session_state)
                st.stop()
        except Exception:
            try:
                st.stop()
            except Exception:
                pass

    # REFRESH => fetch new 10 recommendations
    if st.button("🔄 Refresh App"):
        with st.spinner("Refreshing recommendations..."):
            try:
                prev = st.session_state.get("user_recs", []) or []
                prev_ids = set()
                for it in prev:
                    try:
                        vid = (it.get("id") or {}).get("videoId") or it.get("video_id")
                        if vid:
                            prev_ids.add(str(vid))
                    except Exception:
                        continue

                # fetch primary recommendations (catalog-aware)
                try:
                    df_check = load_catalog()
                    if df_check is None or df_check.empty:
                        fetched = get_any_10()
                    else:
                        fetched = recommend_for_user(st.session_state.get("user_id", "me"), top_k=10)
                except Exception:
                    fetched = get_any_10()

                # normalize helper
                def _normalize(items):
                    out = []
                    for r in items:
                        if not isinstance(r, dict):
                            continue
                        if "id" in r and (r.get("id") or {}).get("videoId"):
                            out.append(r)
                        elif "video_id" in r:
                            out.append({"id": {"videoId": str(r.get("video_id"))}, "snippet": {"title": r.get("title", "") or "", "channelTitle": r.get("channel", "") or ""}})
                    return out

                normalized = _normalize(fetched)

                # If some overlap with previous, try to fill remaining from get_any_10 pool
                unique = []
                seen = set()
                for it in normalized:
                    vid = (it.get("id") or {}).get("videoId")
                    if not vid: continue
                    if vid in prev_ids: continue
                    if vid in seen: continue
                    unique.append(it); seen.add(vid)
                    if len(unique) >= 10:
                        break

                if len(unique) < 10:
                    # try extra candidates to fill
                    try:
                        extras = get_any_10()
                    except Exception:
                        extras = []
                    for ex in _normalize(extras):
                        vid = (ex.get("id") or {}).get("videoId")
                        if not vid: continue
                        if vid in prev_ids or vid in seen: continue
                        unique.append(ex); seen.add(vid)
                        if len(unique) >= 10:
                            break

                final = unique
                if not final:
                    # fallback to normalized (allow repeats if nothing else)
                    final = normalized[:10]

                st.session_state["user_recs"] = final[:10]
                st.success(f"✅ Refreshed: {len(st.session_state['user_recs'])} songs")
            except Exception as e:
                st.error(f"Refresh failed: {str(e)[:200]}")
        # rerun only after the button action completes to reflect updated session_state
        _safe_rerun()

    if st.button("🎵 Recommend for me"):
        with st.spinner("Computing 10 recommendations..."):
            try:
                recs = recommend_for_user(st.session_state.get("user_id", "me"), top_k=10)
                normalized = []
                for r in recs:
                    if not isinstance(r, dict):
                        continue
                    if "id" in r and (r.get("id") or {}).get("videoId"):
                        normalized.append(r)
                    elif "video_id" in r:
                        normalized.append({"id": {"videoId": str(r.get("video_id"))}, "snippet": {"title": r.get("title", "") or "", "channelTitle": r.get("channel", "")}})
                st.session_state["user_recs"] = normalized[:10]
                st.success(f"✅ Found {len(st.session_state['user_recs'])} recommendations!")
            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")

    recs = st.session_state.get("user_recs") or []
    if not recs:
        st.info("Click '🎵 Recommend for me' to get started!")
    else:
        st.caption(f"Found {len(recs)} songs")
        for i, row in enumerate(recs, 1):
            snippet = row.get("snippet", {}) if isinstance(row, dict) else {}
            title = snippet.get("title") or row.get("title", row.get("video_id", "Unknown"))
            channel = snippet.get("channelTitle") or row.get("channel", "Unknown")
            vid = (row.get("id") or {}).get("videoId") or row.get("video_id")
    
            render_song_card(i, vid, title, channel)
            
            if st.button("💾 Save", key=f"save_user_{vid}"):
                save_to_library(video_id=vid, title=title, user_id=st.session_state.get("user_id", "me"))

    st.markdown("---")
    st.subheader("Your Listening History")
    history = load_history(st.session_state.get("user_id", "me")) or []
    if not history:
        st.info("No history yet. Save songs to build your history!")
    else:
        st.caption(f"Total: {len(history)} songs | Showing last 10")
        df = load_catalog()
        for j, video_id in enumerate(reversed(history[-10:]), 1):
            row_info = None
            if not df.empty and "video_id" in df.columns:
                matches = df[df["video_id"].astype(str) == str(video_id)]
                if not matches.empty:
                    row_info = matches.iloc[0].to_dict()
            title = (row_info.get("title") if row_info else video_id) or video_id
            channel = (row_info.get("channel") if row_info else "") or ""
            
            render_song_card(j, video_id, title, channel)

# ============================================================================
# ARTIST MODE
# ============================================================================
else:
    st.subheader(f"Discover {st.session_state.get('artist', 'Music')}")
    st.caption(f"Model: {st.session_state.get('artist_model', 'Hybrid')}")

    if st.button("🎵 Recommend 10 Songs"):
        with st.spinner("Fetching 10 songs..."):
            try:
                artist = st.session_state.get("artist", ARTISTS[0])
                videos = get_exactly_10(artist)

                while len(videos) < 10:
                    extras = get_exactly_10(random.choice(ARTISTS))
                    existing = set((v.get("id") or {}).get("videoId") for v in videos)
                    for v in extras:
                        vid_new = (v.get("id") or {}).get("videoId")
                        if vid_new and vid_new not in existing:
                            videos.append(v)
                            existing.add(vid_new)
                        if len(videos) >= 10:
                            break

                st.session_state["artist_results"] = videos[:10]
                st.success(f"✅ Found {len(st.session_state['artist_results'])} songs!")
            except Exception as e:
                st.error(f"Error fetching songs: {str(e)[:200]}")

    artist_results = st.session_state.get("artist_results", [])
    if not artist_results:
        st.info("Click '🎵 Recommend 10 Songs' to discover music!")
    else:
        st.caption(f"Showing {len(artist_results)} songs")
        for i, v in enumerate(artist_results, 1):
            snippet = v.get("snippet", {})
            title = snippet.get("title", "Unknown Title")
            channel = snippet.get("channelTitle", "Unknown Artist")
            vid = (v.get("id", {}) or {}).get("videoId")
            
            if not vid:
                continue

            render_song_card(i, vid, title, channel)
            
            if st.button("💾 Save", key=f"save_artist_{vid}_{i}"):
                save_to_library(video_id=vid, title=title, user_id=get_active_user_id())