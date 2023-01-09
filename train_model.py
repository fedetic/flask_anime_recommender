import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import pickle

# Load the data
animes = pd.read_csv("animes.csv")
profiles = pd.read_csv("profiles.csv")
reviews = pd.read_csv("reviews.csv")

# Preprocessing the data
# Extract relevant features from the animes and profiles data
anime_features = animes[["genre", "popularity", "score"]]
user_features = profiles[["favorites_anime"]]

# Create a matrix of user-anime ratings
ratings_matrix = reviews.pivot(index="profile_id", columns="anime_id", values="score")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(anime_features, ratings_matrix, test_size=0.2)

# Train the model
model = TruncatedSVD(n_components=50)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
print(f"RMSE: {rmse:.2f}")

# Save the model to a pickle file
with open("recommender_model.pkl", "wb") as f:
    pickle.dump(model, f)
