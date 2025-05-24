import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Generate random dataset of 150 people
np.random.seed(42)

n_samples = 150

data = {
    "Age": np.random.randint(18, 60, size=n_samples),
    "Gender": np.random.choice(["Male", "Female"], size=n_samples),
    "Role": np.random.choice(["student", "employee", "private owner", "None"], size=n_samples),
    "Activity_Level": np.random.choice(["Low", "Medium", "High"], size=n_samples),
    "Diet": np.random.choice(["Vegetarian", "Non-Vegetarian", "Vegan"], size=n_samples),
    "Fitness_Goal": np.random.choice(["Weight Loss", "Muscle Gain", "Maintenance"], size=n_samples),
    "Health_Condition": np.random.choice(["Good", "Average", "Poor"], size=n_samples),
    # Targets:
    "Height_cm": np.random.randint(140, 200, size=n_samples),
    "Weight_kg": np.random.randint(45, 120, size=n_samples),
}

df = pd.DataFrame(data)

# Features and targets
X = df[["Age", "Gender", "Role", "Activity_Level", "Diet", "Fitness_Goal", "Health_Condition"]].copy()  # <-- copy here
y = df[["Height_cm", "Weight_kg"]]

# Encode categorical features with .loc to avoid warnings
le_gender = LabelEncoder()
le_role = LabelEncoder()
le_activity = LabelEncoder()
le_diet = LabelEncoder()
le_fitness = LabelEncoder()
le_health = LabelEncoder()

X.loc[:, "Gender_enc"] = le_gender.fit_transform(X["Gender"])
X.loc[:, "Role_enc"] = le_role.fit_transform(X["Role"])
X.loc[:, "Activity_enc"] = le_activity.fit_transform(X["Activity_Level"])
X.loc[:, "Diet_enc"] = le_diet.fit_transform(X["Diet"])
X.loc[:, "Fitness_enc"] = le_fitness.fit_transform(X["Fitness_Goal"])
X.loc[:, "Health_enc"] = le_health.fit_transform(X["Health_Condition"])

X_final = X[["Age", "Gender_enc", "Role_enc", "Activity_enc", "Diet_enc", "Fitness_enc", "Health_enc"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Train multi-output model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_scaled, y)

# Create model folder if not exist
if not os.path.exists("model"):
    os.makedirs("model")

# Save all objects
joblib.dump(model, "model/multioutput_rf_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le_gender, "model/le_gender.pkl")
joblib.dump(le_role, "model/le_role.pkl")
joblib.dump(le_activity, "model/le_activity.pkl")
joblib.dump(le_diet, "model/le_diet.pkl")
joblib.dump(le_fitness, "model/le_fitness.pkl")
joblib.dump(le_health, "model/le_health.pkl")

print("Training complete, all models and encoders saved in 'model/' folder.")
