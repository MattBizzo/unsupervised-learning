import pandas as pd
from sklearn.preprocessing import StandardScaler

movies = pd.read_csv("movies.csv")
genres = movies.genres.str.get_dummies()

movies_data = pd.concat([movies, genres], axis=1)

scaler = StandardScaler()
scaled_genres = scaler.fit_transform(genres)

print(scaled_genres.shape)
