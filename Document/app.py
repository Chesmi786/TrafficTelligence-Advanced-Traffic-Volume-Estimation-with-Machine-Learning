from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        holiday = request.form["holiday"]
        weather = request.form["weather"]
        time_of_day = request.form["time"]
        temp = float(request.form["temp"])
        rain = float(request.form["rain"])
        snow = float(request.form["snow"])

        # Transform categorical inputs
        holiday_encoded = encoders["holiday"].transform([holiday])[0]
        weather_encoded = encoders["weather"].transform([weather])[0]
        time_encoded = encoders["time"].transform([time_of_day])[0]

        X = pd.DataFrame([{
            "temp": temp,
            "rain": rain,
            "snow": snow,
            "holiday_encoded": holiday_encoded,
            "weather_encoded": weather_encoded,
            "time_encoded": time_encoded
        }])

        prediction = model.predict(X)[0]
        prediction = round(prediction, 2)

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
