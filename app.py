from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and encoders once on app start
model = joblib.load("model/multioutput_rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le_gender = joblib.load("model/le_gender.pkl")
le_role = joblib.load("model/le_role.pkl")
le_activity = joblib.load("model/le_activity.pkl")
le_diet = joblib.load("model/le_diet.pkl")
le_fitness = joblib.load("model/le_fitness.pkl")
le_health = joblib.load("model/le_health.pkl")

def safe_transform(le, label):
    """Encode label using LabelEncoder if known; else raise error."""
    if label in le.classes_:
        return le.transform([label])[0]
    else:
        raise ValueError(f"Unexpected label '{label}' for encoder with classes {list(le.classes_)}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            gender = request.form["gender"]
            role = request.form["role"]
            activity = request.form["activity"]
            diet = request.form["diet"]
            fitness = request.form["fitness"]
            health = request.form["health"]

            # Now the input labels exactly match the training labels, so safe_transform will work
            gender_enc = safe_transform(le_gender, gender)
            role_enc = safe_transform(le_role, role)
            activity_enc = safe_transform(le_activity, activity)
            diet_enc = safe_transform(le_diet, diet)
            fitness_enc = safe_transform(le_fitness, fitness)
            health_enc = safe_transform(le_health, health)

            features = np.array([[age, gender_enc, role_enc, activity_enc, diet_enc, fitness_enc, health_enc]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            height_pred = round(prediction[0], 2)
            weight_pred = round(prediction[1], 2)

            return render_template("index.html", height=height_pred, weight=weight_pred, submitted=True)

        except ValueError as ve:
            return f"Encoding error: {ve}"

    return render_template("index.html", submitted=False)
if __name__ == "__main__":
    app.run(debug=True)

