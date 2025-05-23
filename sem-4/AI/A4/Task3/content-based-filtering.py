import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load ratings:
ratings_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
ratings = pd.read_csv(ratings_url, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

# Normalizing ratings:
ratings["rating"] = ratings["rating"] / 5.0

# Load movie genres:
movies_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movies = pd.read_csv(movies_url, sep="|", encoding="latin-1", header=None,
                     names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
                            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])

# Extract genre features:
genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
              "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
              "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
genre_matrix = movies[genre_cols]
genre_matrix.index = movies["movie_id"]

# Split data into train/test sets:
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# Building user profile vectors from training set:
user_profiles = {}
for user_id in train_ratings["user_id"].unique():
    user_data = train_ratings[train_ratings["user_id"] == user_id]
    liked = user_data[user_data["rating"] >= 4]["item_id"]
    liked_genres = genre_matrix.loc[genre_matrix.index.intersection(liked)]
    
    if not liked_genres.empty:
        user_profiles[user_id] = liked_genres.mean().values
    else:
        user_profiles[user_id] = np.zeros(len(genre_cols))

# Predict on test set using cosine similarity:
y_true = []
y_pred = []
for _, row in test_ratings.iterrows():
    user_id = row["user_id"]
    movie_id = row["item_id"]
    
    if user_id in user_profiles and movie_id in genre_matrix.index:
        user_vec = user_profiles[user_id].reshape(1, -1)
        movie_vec = genre_matrix.loc[movie_id].values.reshape(1, -1)
        similarity = cosine_similarity(user_vec, movie_vec)[0][0]
        
        # Scale similarity to 1–5 range
        predicted_rating = similarity * 5
        y_pred.append(predicted_rating)
        y_true.append(row["rating"])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Content-Based Filtering RMSE (Cosine Similarity): {rmse:.4f}")
