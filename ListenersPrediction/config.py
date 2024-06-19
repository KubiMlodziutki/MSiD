#config
CLIENT_ID = "put your client id here"
CLIENT_SECRET = "put your client secret here"
SAVE_FILE = "spotify_data7.json"
READ_FILE = "spotify_song_database.json"
DOWN_FOLLOWERS = 35000
UPPER_FOLLOWERS = 1500000
DOWN_LOUDNESS = -15
UPPER_LOUDNESS = -2
DOWN_DANCEABILITY = 0.25
DOWN_POPULARITY = 3
UPPER_POPULARITY = 75

INFO_MESSAGE = (
            "Hello!\n"
            "This is an app to predict your song's popularity based on:\n"
            f"Loudness (Best for values between {DOWN_LOUDNESS} and {UPPER_LOUDNESS})\n"
            f"Followers (Best for values between {DOWN_FOLLOWERS} and {UPPER_FOLLOWERS})\n"
            f"Danceability (Best for values over {DOWN_DANCEABILITY} to max {1.0})\n\n"
            f"Approximated accuracies of available models:\n"
            f"Linear Regression - 40%\n"
            f"Random Forest - 60%\n\n"
            f"To modify train dataset - check spotify_song_database.json file\n\n"
            f"Any changes in dataset, config (boundaries) or analyzer.py file may cause errors (or improvements)"
        )

# to expand / shorten
GENRES = [
    "pop",
    "guitar",
    "hip-hop",
    "holidays",
    "house",
    "j-pop",
    "k-pop",
    "kids",
    "latin",
    "metal",
    "movies",
    "new-age",
    "new-release",
    "party",
    "punk",
    "reggae",
    "rock",
    "rockabilly",
    "romance",
    "spanish",
    "study",
    "summer",
  ]