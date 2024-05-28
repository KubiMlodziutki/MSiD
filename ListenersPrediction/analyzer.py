import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

dataframe = pd.read_json('spotify_data4.json')

print(dataframe.sample(n=10))
print()
print(dataframe.describe())

dataframe.drop_duplicates(inplace=True)

# Scatter plot: Popularity vs Followers
plt.figure(figsize=(10, 6))
sns.scatterplot(x='followers', y='popularity', data=dataframe)
plt.xlabel('Followers')
plt.ylabel('Popularity')
plt.title('Popularity vs Followers')
plt.tight_layout()
plt.show()

# Scatter plot: Popularity vs Danceability
plt.figure(figsize=(10, 6))
sns.scatterplot(x='danceability', y='popularity', data=dataframe)
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.title('Popularity vs Danceability')
plt.tight_layout()
plt.show()

# Scatter plot: Popularity vs Loudness
plt.figure(figsize=(10, 6))
sns.scatterplot(x='loudness', y='popularity', data=dataframe)
plt.xlabel('Loudness')
plt.ylabel('Popularity')
plt.title('Popularity vs Loudness')
plt.tight_layout()
plt.show()

# Scatter plot: Popularity vs Key
plt.figure(figsize=(10, 6))
sns.scatterplot(x='key', y='popularity', data=dataframe)
plt.xlabel('Key')
plt.ylabel('Popularity')
plt.title('Popularity vs Key')
plt.tight_layout()
plt.show()

# Histogram: Distribution of Popularity
plt.figure(figsize=(10, 6))
sns.histplot(dataframe['popularity'], kde=True, bins=10, color='green')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of Popularity')
plt.tight_layout()
plt.show()

# Boxplot: Popularity distribution by Genre
plt.figure(figsize=(10, 6))
sns.boxplot(x='track_genre', y='popularity', data=dataframe)
plt.xlabel('Track Genre')
plt.ylabel('Popularity')
plt.title('Popularity Distribution by Genre')
plt.tight_layout()
plt.show()

# Violin Plot: Popularity distribution by Artist
plt.figure(figsize=(10, 6))
sns.violinplot(x='artist', y='popularity', data=dataframe)
plt.xlabel('Artist')
plt.ylabel('Popularity')
plt.title('Popularity Distribution by Artist')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pairplot to see all relationships
# sns.pairplot(dataframe, x_vars=['followers', 'danceability', 'loudness', 'key'], y_vars='popularity', height=5, aspect=0.8)
# plt.suptitle('Pairplot of Features vs Popularity', y=1.02)
# plt.show()