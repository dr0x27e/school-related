# collaborative_filtering.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
import matplotlib.pyplot as plt

# Load ratings
df = pd.read_csv(
    "https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)

n_movies = df["item_id"].nunique()
n_users  = df["user_id"].nunique()

print(f"\nUsers: {n_users}\nMovies: {n_movies}\n")

# Label encode user and movie IDs
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
df["user_enc"] = user_encoder.fit_transform(df["user_id"])
df["movie_enc"] = movie_encoder.fit_transform(df["item_id"])

X = df[["user_enc", "movie_enc"]]
y = df["rating"] / 5.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((
    {'user': X_train["user_enc"].values, 'movie': X_train["movie_enc"].values},
    y_train.values.astype('float32')
)).shuffle(1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {'user': X_test["user_enc"].values, 'movie': X_test["movie_enc"].values},
    y_test.values.astype('float32')
)).batch(64)

# Collaborative model
embedding_dim = 32
user_input = Input(shape=(1,), name='user')
movie_input = Input(shape=(1,), name='movie')

user_emb = Embedding(len(user_encoder.classes_), embedding_dim)(user_input)
movie_emb = Embedding(len(movie_encoder.classes_), embedding_dim)(movie_input)

user_vec = Flatten()(user_emb)
movie_vec = Flatten()(movie_emb)

x = tf.keras.layers.Concatenate()([user_vec, movie_vec])
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)

# Evaluate
y_pred = model.predict(test_dataset)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Collaborative Filtering RMSE: {rmse:.4f}")

# Plot
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Collaborative Filtering Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
