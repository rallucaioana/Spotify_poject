import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

artists=pd.read_csv("artist_data.csv")

#visualize data
artists.head()

#columns in the data
col= artists.columns
col

#datatypes
artists.dtypes

#unique artists
unique=[]
for a in artists["name"]:
    if a not in unique:
        unique.append(a)
len(unique)

#top 10 artists by popularity
top_popularity=artists.sort_values(by="artist_popularity", ascending=False).head(10)
plt.plot(top_popularity["name"],top_popularity["artist_popularity"])
plt.title("Top 10 Artists by Popularity")
plt.ylabel("Popularity")
plt.tight_layout()
plt.show()

#top 10 artists by nr of followers
top_followers=artists.sort_values(by="followers", ascending=False).head(10)
plt.plot(top_followers["name"],top_followers["followers"])
plt.title("Top 10 Artists by Follwers")
plt.ylabel("Log(Followers)")
plt.tight_layout()
plt.show()

#scale it bcs nr of followers are huge compared to popularity and grow exponentially
artists["log_followers"] = np.log(artists["followers"] + 1)

plt.figure()
plt.scatter(artists["log_followers"], artists["artist_popularity"], alpha=0.5)
plt.xlabel("Log(Followers)")
plt.ylabel("Popularity")
plt.title("Popularity vs Followers")
plt.show()


# Correlation between popularity and log(followers)
corr = artists["artist_popularity"].corr(artists["log_followers"])
print(corr)

#top artists by genre
def top_artists_by_genre(genre):
    genre = genre.lower()
    genre_filter = [genre in str(g).lower() for g in artists["artist_genres"]]
    filtered = artists[genre_filter]
    return filtered.sort_values(by="artist_popularity", ascending=False).head(10)

genre_cols = ["genre_0", "genre_1", "genre_2", "genre_3",
              "genre_4", "genre_5", "genre_6"]
artists["num_genres"] = artists[genre_cols].count(axis=1)


plt.scatter(artists["num_genres"], artists["artist_popularity"])
plt.xlabel("Number of Genres")
plt.ylabel("Popularity")
plt.title("Number of Genres vs Popularity")
plt.show()

plt.scatter(artists["num_genres"], artists["followers"])
plt.xlabel("Number of Genres")
plt.ylabel("Followers")
plt.title("Number of Genres vs Followers")
plt.show()