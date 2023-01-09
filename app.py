from flask import Flask, request, render_template
import pickle

# Load the saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get the user's favorite anime from the request
    favorite_anime = request.form["favorite_anime"]
    
    # Use the model to make recommendations
    recommendations = model.predict(favorite_anime)
    
    # Render the recommendations template and pass the recommendations to it
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run()
