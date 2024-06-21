import requests
import spotipy
import json
import time
from spotipy.oauth2 import SpotifyClientCredentials
from config import CLIENT_SECRET, CLIENT_ID, SAVE_FILE, GENRES
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TrackData:
    track_id: str
    track_name: str
    track_genre: str
    artist: str
    followers: int
    country: str
    popularity: int
    danceability: float
    loudness: float
    key: int


def get_spotify_client() -> spotipy.Spotify:
    credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return spotipy.Spotify(client_credentials_manager=credentials_manager)


# to be used or not to be used
def get_artist_country(artist_name: str) -> str:
    url = f"https://musicbrainz.org/ws/2/artist/?query={artist_name}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        artists = data.get('artists', [])
        if artists:
            return artists[0].get('country', 'Unknown')

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")

    except json.decoder.JSONDecodeError:
        print("Invalid JSON in response")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return 'Unknown'


def save_data_to_json(data: List[Dict[str, Any]], filename: str = SAVE_FILE) -> None:
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


# fetching data and putting it into database
def fetch_data(query: str, spotify: spotipy.Spotify, total_limit: int = 300) -> List[TrackData]:
    data = []
    limit = 15
    for offset in range(0, total_limit, limit):
        try:
            time.sleep(32)
            results = spotify.search(q=query, type='track', limit=limit, offset=offset)
            tracks = results['tracks']['items']

            for track in tracks:
                time.sleep(5)
                track_id = track['id']
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                artist_id = track['artists'][0]['id']
                artist_info = spotify.artist(artist_id)
                track_genre = query
                country = 'Unknown'  # temporary
                followers = artist_info['followers']['total']
                track_popularity = track['popularity']
                audio_features = spotify.audio_features(track_id)[0]

                track_data = TrackData(
                    track_id=track_id,
                    track_name=track_name,
                    track_genre=track_genre,
                    artist=artist_name,
                    followers=followers,
                    country=country,
                    popularity=track_popularity,
                    danceability=audio_features.get('danceability', 0) if audio_features else 0,
                    loudness=audio_features.get('loudness', -60) if audio_features else -60,
                    key=audio_features.get('key', 0) if audio_features else 0
                )
                data.append(track_data)
                all_data.append(track_data)

            save_data_to_json([track.__dict__ for track in all_data])

        except Exception as e:
            print(f"Error occurred: {e}")
            save_data_to_json([track.__dict__ for track in all_data])
            break

        if len(tracks) < limit:
            break

    return data


all_data: List[TrackData] = []
if __name__ == "__main__":
    spotify_client = get_spotify_client()
    try:
        for genre in GENRES:
            print(f"Starting data collection for genre: {genre}")
            genre_data = fetch_data(genre, spotify_client)
            print(f"Collected {len(genre_data)} records for genre: {genre}")

    except Exception as e:
        print(f"An error occurred during data collection: {e}")

    finally:
        save_data_to_json([track.__dict__ for track in all_data])
