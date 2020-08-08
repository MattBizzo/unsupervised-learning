import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

movies = pd.read_csv("movies.csv")
# get categorical var genres column and convert into dummy/indicator vars
genres = movies.genres.str.get_dummies()

movies_data = pd.concat([movies, genres], axis=1)

scaler = StandardScaler()
scaled_genres = scaler.fit_transform(genres)

model = KMeans(n_clusters=5)
model.fit(scaled_genres)

print(f"Clusters {model.labels_}")
