import base64
import requests
import streamlit as st


TOKEN_URL = "https://accounts.spotify.com/api/token"
BASE_URL = "https://api.spotify.com/v1"


@st.cache_data(ttl=300)
def get_spotify_access_token():
    client_id = st.secrets["spotify"]["client_id"]
    client_secret = st.secrets["spotify"]["client_secret"]

    auth_string = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

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


# caches loaded album covers for 24 hours
@st.cache_data(ttl=86400)
def get_album_cover_data(album_id):
    token = get_spotify_access_token()

    response = requests.get(
        f"{BASE_URL}/albums/{album_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()

    images = data.get("images", [])
    image_url = images[0]["url"] if images else None
    spotify_url = data.get('external_urls', {}).get("spotify")

    return {
        "image_url": image_url,
        "spotify_url": spotify_url,
        "album_name": data.get("name"),
    }