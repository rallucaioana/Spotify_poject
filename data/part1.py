import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv("artist_data.csv")

print("Columns:")
for col in df.columns:
    print(col)


unique_artists = df["name"].nunique()
print("Number of unique artists:", unique_artists)

# Top 10 by popularity
top_popularity = df.sort_values("artist_popularity", ascending=False).head(10)

plt.figure()
plt.bar(top_popularity["name"], top_popularity["artist_popularity"])
plt.title("Top 10 Artists by Popularity")
plt.show()

# Top 10 by followers
top_followers = df.sort_values("followers", ascending=False).head(10)

plt.figure()
plt.bar(top_followers["name"], top_followers["followers"])
plt.title("Top 10 Artists by Followers")
plt.show()


# Correlation
correlation = df["artist_popularity"].corr(df["followers"])
print(f"Correlation between popularity and followers:", correlation)

df["log_followers"] = np.log(df["followers"] + 1)
plt.figure()
plt.scatter(df["log_followers"], df["artist_popularity"], alpha=0.5)
plt.xlabel("log(Followers)")
plt.ylabel("Artist Popularity")
plt.title("Popularity vs log(Followers)")
plt.show()

# Linear regression
X = sm.add_constant(df["log_followers"])
y = df["artist_popularity"]
model = sm.OLS(y, X).fit()
print(model.summary())

df["predicted_popularity"] = model.predict(X)
df["residuals"] = df["artist_popularity"] - df["predicted_popularity"]

overperformers = df.sort_values("residuals", ascending=False).head(10)

print("\n Overperformers:")
print(overperformers[["name", "artist_popularity", "followers"]])

legacy = df.sort_values("residuals").head(10)

print("\nLegacy Artists:")
print(legacy[["name", "artist_popularity", "followers"]])

# relevance of genres
genre_columns = [col for col in df.columns if "genre_" in col]

df["num_genres"] = df[genre_columns].notna().sum(axis=1)
plt.figure()
df.groupby("num_genres")["artist_popularity"].mean().plot(kind="bar")
plt.xlabel("Number of Genres")
plt.ylabel("Average Popularity")
plt.title("Average Popularity by Number of Genres")
plt.show()

print("\nCorrelation (genres vs popularity):",
      df["num_genres"].corr(df["artist_popularity"]))

print("Correlation (genres vs followers):",
      df["num_genres"].corr(df["followers"]))


def top_ten_by_genre(genre_name):
    mask = df[genre_columns].apply(lambda row: genre_name in row.values, axis=1)
    return df[mask].sort_values("artist_popularity", ascending=False).head(10)

print("\nTop 10 Cello Artists:")
print(top_ten_by_genre("cello")[["name", "artist_popularity"]])
