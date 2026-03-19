

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv("/Users/samira/Desktop/Spotify/artist_data.csv")

print("\nColumns in dataset:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nNumber of unique artists:", df["name"].nunique())


top_popularity = df.sort_values("artist_popularity", ascending=False).head(10)
top_followers = df.sort_values("followers", ascending=False).head(10)

# Plot Top 10 by Popularity
plt.figure()
sns.barplot(data=top_popularity, x="artist_popularity", y="name")
plt.title("Top 10 Artists by Popularity")
plt.tight_layout()
plt.show()

# Plot Top 10 by Followers
plt.figure()
sns.barplot(data=top_followers, x="followers", y="name")
plt.title("Top 10 Artists by Followers")
plt.tight_layout()
plt.show()

#Popularity vs Followers

df["log_followers"] = np.log(df["followers"] + 1)

correlation = df["artist_popularity"].corr(df["log_followers"])
print("\nCorrelation between popularity and log(followers):", correlation)

# Scatter plot
plt.figure()
sns.scatterplot(data=df, x="log_followers", y="artist_popularity")
plt.title("Popularity vs Log(Followers)")
plt.show()


X = sm.add_constant(df["log_followers"])
y = df["artist_popularity"]

model = sm.OLS(y, X).fit()
print(model.summary())

df["predicted_popularity"] = model.predict(X)
df["residuals"] = df["artist_popularity"] - df["predicted_popularity"]

#Over-performers  
over_performers = df.sort_values("residuals", ascending=False).head(10)

#Legacy artists
legacy_artists = df.sort_values("residuals").head(10)

print("\nTop Over-Performers:")
print(over_performers[["name", "artist_popularity", "followers"]])

print("\nLegacy Artists:")
print(legacy_artists[["name", "artist_popularity", "followers"]])

#genre analysis

def top_10_by_genre(genre_name):
    genre_df = df[df["artist_genres"].str.contains(genre_name, case=False, na=False)]
    return genre_df.sort_values("artist_popularity", ascending=False).head(10)

print("\nTop 10 Pop Artists:")
print(top_10_by_genre("pop")[["name", "artist_popularity"]])

#count number of genres

genre_columns = ["genre_0", "genre_1", "genre_2",
                 "genre_3", "genre_4", "genre_5", "genre_6"]

df["num_genres"] = df[genre_columns].notna().sum(axis=1)

# Distribution plot
plt.figure()
sns.histplot(df["num_genres"], bins=10)
plt.title("Distribution of Number of Genres per Artist")
plt.show()

#Correlations
print("\nCorrelation between number of genres and popularity:",
      df["num_genres"].corr(df["artist_popularity"]))

print("Correlation between number of genres and followers:",
      df["num_genres"].corr(df["followers"]))

#Scatter plot
plt.figure()
sns.scatterplot(data=df, x="num_genres", y="artist_popularity")
plt.title("Number of Genres vs Popularity")
plt.show()

#creative insight
plt.figure()
sns.histplot(df["artist_popularity"], bins=20)
plt.title("Distribution of Artist Popularity")
plt.show()

plt.figure()
sns.histplot(df["log_followers"], bins=20)
plt.title("Distribution of Log(Followers)")
plt.show()

print("\nAnalysis complete.")

