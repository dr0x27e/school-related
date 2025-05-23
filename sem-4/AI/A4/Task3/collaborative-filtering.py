import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# Load and Preprocess the Dataset
df = pd.read_csv(
    "https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)

n_users = df["user_id"].nunique()
n_movies = df["item_id"].nunique()
print(f"Number of unique users: {n_users}")
print(f"Number of unique movies: {n_movies}")

# Label encode user and movie ids (might not be needed):
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
df["user_enc"] = user_encoder.fit_transform(df["user_id"])
df["movie_enc"] = movie_encoder.fit_transform(df["item_id"])

# Create user-item matrix using Pandas pivot:
user_item_matrix = df.pivot(index="user_enc", columns="movie_enc", values="rating")
# Fill NaN with 0 for reference:
user_item_matrix = user_item_matrix.fillna(0)
print(f"User-item matrix shape: {user_item_matrix.shape}")
print(f"Sparsity: {1 - (df['rating'].count() / (n_users * n_movies)):.4%}")

# Normalize
y = df["rating"] / 5.0

# Prepare data for the model:
X = df[["user_enc", "movie_enc"]]

# Split the dataset into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TensorFlow datasets:
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"user": X_train["user_enc"].values, "movie": X_train["movie_enc"].values},
    y_train.values.astype("float32")
)).shuffle(1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {"user": X_test["user_enc"].values, "movie": X_test["movie_enc"].values},
    y_test.values.astype("float32")
)).batch(64)

# Build Matrix Factorization Model using Neural Network:
embedding_dim = 32

# Input layers:
user_input = Input(shape=(1,), name="user")
movie_input = Input(shape=(1,), name="movie")

# Embedding layers:
user_emb = Embedding(input_dim=n_users, output_dim=embedding_dim, name="user_embedding")(user_input)
movie_emb = Embedding(input_dim=n_movies, output_dim=embedding_dim, name="movie_embedding")(movie_input)

# Flatten embeddings:
user_vec = Flatten()(user_emb)
movie_vec = Flatten()(movie_emb)

# Concatenate user and movie vectors:
concat = Concatenate()([user_vec, movie_vec])

# hidden layers with relu activation:
x = Dense(128, activation="relu")(concat)
x = Dense(64, activation="relu")(x)

# Output layer with sigmoid activation since we scale the input:
output = Dense(1, activation="sigmoid")(x)

# Creating and compile the model:
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Display model summary:
model.summary()

# Training the model:
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    verbose=1
)

# Predict ratings:
y_pred = model.predict(test_dataset)

# Rescale predictions:
y_pred_rescaled = y_pred * 5.0
y_test_rescaled = y_test.values * 5.0

# RMSE:
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
