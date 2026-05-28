#     Bert+ Linear Layer
Not Performing Well
#     TF-IDF + LR
Training samples: 325476 | Testing samples: 81370
Converting text to TF-IDF features...
Training Ridge Linear Regression model...

--- Model Evaluation ---
Mean Squared Error: 34062.23
Mean Absolute Error (Average Kcal off by): 141.12 Kcal
R-squared (R2) Score: 0.3080

--- Top Ingredient Indicators ---

Top 5 High-Calorie Indicators:
     Word     Weight
 chickens 872.948336
spareribs 807.339457
   pectin 682.369240
 linguine 660.774521
    fryer 571.073979

Top 5 Low-Calorie Indicators:
    Word      Weight
   broth -387.250114
  simply -321.441018
 carcass -267.095170
brussels -252.753218
   panir -243.758570

#     For Linear Reg cal pred if as features fats prot carbs taken it best.
