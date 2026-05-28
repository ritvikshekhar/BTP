import pandas as pd
import torch
import ast
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. LOAD AND CLEAN THE DATASET
# ==========================================
import pandas as pd
import ast

# ==========================================
# 1. LOAD AND CLEAN THE DATASET
# ==========================================
print("Loading dataset...")
df = pd.read_csv("Merged_RecipeDB.csv") 

# --- NEW DATA CLEANING STEPS ---

# 1. Drop rows where either 'Ingredients' OR 'Calories(Kcal)' are missing (NaN/Null)
df = df.dropna(subset=['Ingredients', 'Calories(Kcal)'])

# 2. Keep exactly 5,00 random data points
# We check the length first just in case your dataset shrinks below 5000 after dropping nulls
if len(df) > 5000:
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    print("Successfully sampled 5,000 random rows.")
else:
    print(f"Note: After dropping nulls, only {len(df)} rows remain. Using all available data.")

# -------------------------------

# Now proceed with your existing text cleaning step...
def clean_text(text):
    try:
        items = ast.literal_eval(text)
        return ", ".join(items)
    except:
        return str(text)

df['clean_ingredients'] = df['Ingredients'].apply(clean_text)

# (Continue to Step 2: Train, Validation, and Test Split...)



# ==========================================
# 2. TRAIN, VALIDATION, AND TEST SPLIT
# ==========================================
print("Splitting data into Train, Validation, and Test sets...")

# First, separate 15% for the final TEST set
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

# Next, separate the remaining data into TRAIN and VALIDATION
# 0.1765 of the remaining 85% is roughly 15% of the total dataset
train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)

print(f"Training samples: {len(train_df)} | Validation samples: {len(val_df)} | Testing samples: {len(test_df)}")

# ==========================================
# 3. SETUP BERT TOKENIZER & DATASET
# ==========================================
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

class RecipeDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = float(self.targets[item])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # For regression, labels must be float32
            'labels': torch.tensor(target, dtype=torch.float32) 
        }

# Create PyTorch datasets
train_dataset = RecipeDataset(train_df['clean_ingredients'].to_numpy(), train_df['Calories(Kcal)'].to_numpy(), tokenizer)
val_dataset = RecipeDataset(val_df['clean_ingredients'].to_numpy(), val_df['Calories(Kcal)'].to_numpy(), tokenizer)
test_dataset = RecipeDataset(test_df['clean_ingredients'].to_numpy(), test_df['Calories(Kcal)'].to_numpy(), tokenizer)

# ==========================================
# 4. INITIALIZE THE BERT MODEL
# ==========================================
print("Initializing BERT model for regression...")
# num_labels=1 is the crucial setting that makes this Regression instead of Classification
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# ==========================================
# 5. TRAINING PHASE
# ==========================================
# Define how we measure success
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {"mse": mse, "mae": mae}

training_args = TrainingArguments(
    output_dir='./bert_recipe_model',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",        # <--- NEW WAY
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,      # We train using Train and Val sets
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# ==========================================
# 6. TESTING PHASE (THE FINAL EXAM)
# ==========================================
print("\n--- Final Testing Phase ---")
# Now we test on the data the model has never seen before
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Set Performance:")
print(f"Mean Squared Error: {test_results['eval_mse']:.2f}")
print(f"Mean Absolute Error (Average Kcal off by): {test_results['eval_mae']:.2f} Kcal")

# Save the final model so you can use it later without retraining
trainer.save_model("./final_bert_calorie_predictor")
print("Model saved to ./final_bert_calorie_predictor")
