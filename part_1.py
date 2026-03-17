import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# importing data
artist_data = pd.read_csv("data/artist_data.csv")
df = pd.DataFrame(artist_data)

#visualize data
df.head()

#columns in the data
col= df.columns
col

#datatypes
df.dtypes

#unique artists
unique=[]
for a in df["name"]:
    if a not in unique:
        unique.append(a)
len(unique)


# basic inspection of the data
# Top 10 by popularity
top_popularity = df.sort_values("artist_popularity", ascending=False).head(10)

plt.figure()
plt.bar(top_popularity["name"], top_popularity["artist_popularity"])
plt.xticks(rotation=45)
plt.title("Top 10 Artists by Popularity")
plt.show()

# Top 10 by followers
top_followers = df.sort_values("followers", ascending=False).head(10)

plt.figure()
plt.bar(top_followers["name"], top_followers["followers"])
plt.xticks(rotation=45)
plt.title("Top 10 Artists by Followers")
plt.show()

#scale it bcs nr of followers are huge compared to popularity and grow exponentially
df["log_followers"] = np.log(df["followers"] + 1)

plt.figure()
plt.scatter(df["log_followers"], df["artist_popularity"], alpha=0.5)
plt.xlabel("Log(Followers)")
plt.ylabel("Popularity")
plt.title("Popularity vs Log(Followers)")
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