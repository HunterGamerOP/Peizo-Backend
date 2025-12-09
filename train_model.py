# train_model.py
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load data
df = pd.read_csv("power_tile_data.csv")

# Make sure column names are correct
df.columns = ["voltage", "current_uA", "weight_kg", "step_location", "power_mW"]

# 2. Features & target
X = df[["voltage", "current_uA", "weight_kg", "step_location"]]
y = df["power_mW"]

# 3. Preprocessing: step_location is categorical
categorical_features = ["step_location"]
numeric_features = ["voltage", "current_uA", "weight_kg"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# 4. Model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# 5. Pipeline = preprocessing + model
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)

print("Train R^2:", pipe.score(X_train, y_train))
print("Test R^2:", pipe.score(X_test, y_test))

# 6. Save trained pipeline
joblib.dump(pipe, "piezo_model.pkl")
print("Saved model as piezo_model.pkl")
