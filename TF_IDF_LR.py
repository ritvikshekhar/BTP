import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

# ==========================================
# 1. LOAD AND CLEAN THE DATASET
# ==========================================
print("Loading dataset...")
df = pd.read_csv("Merged_RecipeDB.csv") 

# Drop missing values
df = df.dropna(subset=['Ingredients', 'Calories(Kcal)'])

# Sample 100,000 data points
# if len(df) > 100000:
    # df = df.sample(n=100000, random_state=42).reset_index(drop=True)

# Clean string representation of lists to plain text
def clean_text(text):
    try:
        items = ast.literal_eval(text)
        return ", ".join(items)
    except:
        return str(text)

df['clean_ingredients'] = df['Ingredients'].apply(clean_text)

# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================
# Unlike deep learning, we only need a Train and Test split here
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_ingredients'], 
    df['Calories(Kcal)'], 
    test_size=0.2, 
    random_state=42
)

print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ==========================================
# 3. APPLY TF-IDF VECTORIZATION
# ==========================================
print("Converting text to TF-IDF features...")
# token_pattern allows the vectorizer to catch numbers and fractions (like 1/3)
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=3000)

# Learn vocabulary from training data and transform it into numbers
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the SAME vocabulary
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# 4. TRAIN THE LINEAR REGRESSION MODEL
# ==========================================
print("Training Ridge Linear Regression model...")
# Ridge Regression works best here because it handles highly correlated text features well
model = Ridge(alpha=1.0)
model.fit(X_train_tfidf, y_train)

# ==========================================
# 5. EVALUATE THE MODEL
# ==========================================
print("\n--- Model Evaluation ---")
predictions = model.predict(X_test_tfidf)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2=r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error (Average Kcal off by): {mae:.2f} Kcal")
print(f"R-squared (R2) Score: {r2:.4f}")
# ==========================================
# 6. BONUS: INTERPRET THE MODEL (See what words drive calories)
# ==========================================
print("\n--- Top Ingredient Indicators ---")
# Get the words and their corresponding regression weights
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_

word_weights = pd.DataFrame({'Word': feature_names, 'Weight': coefficients})

# Top words that INCREASE calories the most
print("\nTop 5 High-Calorie Indicators:")
print(word_weights.sort_values(by='Weight', ascending=False).head(5).to_string(index=False))

# Top words that DECREASE calories the most
print("\nTop 5 Low-Calorie Indicators:")
print(word_weights.sort_values(by='Weight', ascending=True).head(5).to_string(index=False))
