import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("recipes.csv")

# Input Features
X = df[[
    'Fat(gm)',
    'Saturated_Fat(gm)',
    'Cholesterol(gm)',
    'Sodium(gm)',
    'Carbohydrates(gm)',
    'Fiber(gm)',
    'Sugar(gm)',
    'Protein(gm)'
]]

# Target
y = df['Calories(Kcal)']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
