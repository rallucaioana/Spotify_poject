import base64
import time
import requests
import streamlit as st


TOKEN_URL = "https://accounts.spotify.com/api/token"
BASE_URL = "https://api.spotify.com/v1"


class SpotifyRateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


# creates the spotify access token
@st.cache_data(ttl=300)
def get_spotify_access_token():
    client_id = st.secrets["spotify"]["client_id"]
    client_secret = st.secrets["spotify"]["client_secret"]

    auth_string = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

    try:
        response = requests.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=20,
        )

        response.raise_for_status()
        return response.json()["access_token"]

    except requests.exceptions.Timeout:
        st.warning("Spotify authentication timed out. Album covers and artist images will be unavailable.", icon=":material/warning:")
        return None

    except requests.exceptions.RequestException as e:
        st.warning(f"Spotify authentication failed: {e}. Album covers and artist images will be unavailable.", icon=":material/warning:")
        return None


# renders a warning whenever a rate limit has been detected
def render_rate_limit_warning():
    if not st.session_state.get("spotify_rate_limited", False):
        return

    rate_limited_at = st.session_state.get("spotify_rate_limited_at")
    retry_after = st.session_state.get("spotify_retry_after", 30)

    # clears the rate limit whenever the specified amount of time has passed.
    if rate_limited_at and (time.time() - rate_limited_at) > retry_after:
        st.session_state["spotify_rate_limited"] = False
        st.session_state.pop("spotify_retry_after", None)
        st.session_state.pop("spotify_rate_limited_at", None)
        return

    remaining = max(0, int(retry_after - (time.time() - rate_limited_at))) if rate_limited_at else None

    message = (
        f"Spotify API rate limit reached. Album covers and artist images may be missing. {remaining} seconds before rate limit is lifted."
        if remaining is not None
        else "Spotify API rate limit reached. Album covers and artist images may be missing."
    )

    st.warning(message, icon=":material/warning:")


def _set_rate_limited(retry_after: int):
    st.session_state["spotify_rate_limited"] = True
    st.session_state["spotify_retry_after"] = retry_after
    st.session_state["spotify_rate_limited_at"] = time.time()


# Pure cached function: 
# raises SpotifyRateLimitError on 429 so the result
# is never cached (st.cache_data does not cache exceptions), keeping the
# cache key as album_id only and leaving session state mutations to the wrapper.
@st.cache_data(ttl=86400)
def _cached_album_cover(album_id: str):
    token = get_spotify_access_token()

    if token is None:
        return {"image_url": None, "spotify_url": None, "album_name": None}

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/albums/{album_id}", headers=headers, timeout=20)

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 30))
        raise SpotifyRateLimitError(retry_after)

    response.raise_for_status()
    data = response.json()
    images = data.get("images", [])

    # waiting to not overload the API
    time.sleep(0.1)

    return {
        "image_url": images[0]["url"] if images else None,
        "spotify_url": data.get("external_urls", {}).get("spotify"),
        "album_name": data.get("name"),
    }


# Non-cached wrapper: handles session state side effects
def get_album_cover_data(album_id: str):
    try:
        return _cached_album_cover(album_id)
    except SpotifyRateLimitError as e:
        _set_rate_limited(e.retry_after)
        return {"image_url": None, "spotify_url": None, "album_name": None}


# Pure cached function — same pattern as album cover
@st.cache_data(ttl=86400)
def _cached_artist_profile(artist_id: str):
    token = get_spotify_access_token()

    if token is None:
        return {"image_url": None, "spotify_url": None, "name": None}

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/artists/{artist_id}", headers=headers, timeout=20)

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 30))
        raise SpotifyRateLimitError(retry_after)

    response.raise_for_status()
    data = response.json()
    images = data.get("images", [])

    return {
        "image_url": images[0]["url"] if images else None,
        "spotify_url": data.get("external_urls", {}).get("spotify"),
        "name": data.get("name"),
    }


# Non-cached wrapper: handles session state side effects
def get_artist_profile_data(artist_id: str):
    try:
        return _cached_artist_profile(artist_id)
    except SpotifyRateLimitError as e:
        _set_rate_limited(e.retry_after)
        return {"image_url": None, "spotify_url": None, "name": None}


# gets album covers for a list of album ids
def get_album_covers_batch(album_ids: tuple):
    covers = {}

    if st.session_state.get("spotify_rate_limited", False):
        return {}

    for aid in album_ids:
        result = get_album_cover_data(aid)
        if st.session_state.get("spotify_rate_limited"):
            covers["__rate_limited__"] = True
            return covers
        covers[aid] = result

    return covers