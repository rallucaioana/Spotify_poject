import base64
import time
import requests
import streamlit as st


TOKEN_URL = "https://accounts.spotify.com/api/token"
BASE_URL = "https://api.spotify.com/v1"


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
        st.warning("Spotify authentication timed out. Album covers will be unavailable.", icon=":material/warning:")
        return None
    
    except requests.exceptions.RequestException as e:
        st.warning(f"Spotify authentication failed: {e}. Album covers will be unavailable.", icon=":material/warning:")
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


# helper function for album cover fetching
def _fetch_album_cover(album_id: str, token: str) -> dict:
    # protects against errors when token could not be requested
    if token is None:
        return {"image_url": None, "spotify_url": None, "album_name": None}
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/albums/{album_id}", headers=headers, timeout=20)

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 30))
        st.session_state["spotify_rate_limited"] = True
        st.session_state["spotify_retry_after"] = retry_after
        st.session_state["spotify_rate_limited_at"] = time.time()
        return {"image_url": None, "spotify_url": None, "album_name": None}

    response.raise_for_status()
    data = response.json()
    images = data.get("images", [])

    return {
        "image_url": images[0]["url"] if images else None,
        "spotify_url": data.get("external_urls", {}).get("spotify"),
        "album_name": data.get("name"),
    }


@st.cache_data(ttl=86400)
def get_album_cover_data(album_id: str):
    token = get_spotify_access_token()
    return _fetch_album_cover(album_id, token)


# helper function for artist image data fetching
def _fetch_artist_profile(artist_id: str, token: str) -> dict:
    # protects against errors when token could not be requested
    if token is None:
        return {"image_url": None, "spotify_url": None, "name": None}
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/artists/{artist_id}", headers=headers, timeout=20)

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 30))
        st.session_state["spotify_rate_limited"] = True
        st.session_state["spotify_retry_after"] = retry_after
        st.session_state["spotify_rate_limited_at"] = time.time()
        return {"image_url": None, "spotify_url": None, "name": None}

    response.raise_for_status()
    data = response.json()
    images = data.get("images", [])

    return {
        "image_url": images[0]["url"] if images else None,
        "spotify_url": data.get("external_urls", {}).get("spotify"),
        "name": data.get("name"),
    }


# gets an artist's profile data and caches them for 24 hours
@st.cache_data(ttl=86400)
def get_artist_profile_data(artist_id: str):
    token = get_spotify_access_token()
    return _fetch_artist_profile(artist_id, token)


# gets album covers for multiple albums using the batch endpoint (up to 20 IDs per request)
# falls back to sequential single requests per chunk if the batch endpoint returns 403
# caches results for 24 hours
@st.cache_data(ttl=86400)
def get_album_covers_batch(album_ids: tuple) -> dict:
    token = get_spotify_access_token()
    covers = {}
    ids = list(album_ids)
    
    # protects against errors when token could not be requested
    if token is None:
        return {}
    
    for i in range(0, len(ids), 20):
        chunk = [str(aid).strip() for aid in ids[i : i + 20]]

        try:
            response = requests.get(
                f"{BASE_URL}/albums",
                headers={"Authorization": f"Bearer {token}"},
                params={"ids": ",".join(chunk)},
                timeout=20,
            )

            # Rate limited, return whatever we have so far
            if response.status_code == 429:                
                retry_after = int(response.headers.get("Retry-After", 30))
                st.session_state["spotify_rate_limited"] = True
                st.session_state["spotify_retry_after"] = retry_after
                st.session_state["spotify_rate_limited_at"] = time.time()
                covers["__rate_limited__"] = True
                return covers

            if response.status_code == 403:
                raise ValueError("batch_unavailable")

            response.raise_for_status()

            for album in response.json().get("albums") or []:
                if album is None:
                    continue
                aid = album.get("id")
                images = album.get("images", [])
                covers[aid] = {
                    "image_url": images[0]["url"] if images else None,
                    "spotify_url": album.get("external_urls", {}).get("spotify"),
                    "album_name": album.get("name"),
                }

            time.sleep(0.5)

        except ValueError:
            for aid in chunk:
                try:
                    result = _fetch_album_cover(aid, token)
                    if st.session_state.get("spotify_rate_limited"):
                        covers["__rate_limited__"] = True
                        return covers
                    covers[aid] = result
                    time.sleep(0.1)
                except Exception:
                    covers[aid] = {
                        "image_url": None,
                        "spotify_url": None,
                        "album_name": None,
                    }

    return covers