# Import necessary libraries
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
print("--- Loading Data ---")
try:
    train_df = pd.read_csv('train_v9rqX0R.csv')
    test_df = pd.read_csv('test_AbJTz2l.csv')
    print("Data loaded successfully.")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Ensure 'train_v9rqX0R.csv' and 'test_AbJTz2l.csv' are in the directory.")
    exit()

# --- 2. Data Cleaning & Imputation ---
print("\n--- Starting Data Cleaning & Imputation ---")

# Combine for preprocessing
train_len = len(train_df)
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Robust Impute 'Item_Weight' (fallback to global mean if item missing)
overall_weight_mean = combined_df['Item_Weight'].mean()
item_avg_weight = combined_df.groupby('Item_Identifier')['Item_Weight'].mean()
missing_weight_mask = combined_df['Item_Weight'].isnull()
combined_df.loc[missing_weight_mask, 'Item_Weight'] = combined_df.loc[missing_weight_mask, 'Item_Identifier'].apply(
    lambda x: item_avg_weight.get(x, overall_weight_mean)
)
print(f"Missing 'Item_Weight' imputed. Remaining missing: {combined_df['Item_Weight'].isnull().sum()}")

# Impute 'Outlet_Size' based on 'Outlet_Type' mode
outlet_size_mode = combined_df.groupby('Outlet_Type')['Outlet_Size'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Medium')
missing_size_mask = combined_df['Outlet_Size'].isnull()
combined_df.loc[missing_size_mask, 'Outlet_Size'] = combined_df.loc[missing_size_mask, 'Outlet_Type'].map(outlet_size_mode)
print(f"Missing 'Outlet_Size' imputed. Remaining missing: {combined_df['Outlet_Size'].isnull().sum()}")

# Standardize 'Item_Fat_Content'
combined_df['Item_Fat_Content'] = combined_df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
print("'Item_Fat_Content' standardized.")

# --- 3. Feature Engineering ---
print("\n--- Starting Feature Engineering ---")

# Robust Impute zero 'Item_Visibility' (fallback to global mean)
overall_visibility_mean = combined_df['Item_Visibility'].mean()
item_visibility_mean = combined_df.groupby('Item_Identifier')['Item_Visibility'].mean()
zero_visibility_mask = (combined_df['Item_Visibility'] == 0)
combined_df.loc[zero_visibility_mask, 'Item_Visibility'] = combined_df.loc[zero_visibility_mask, 'Item_Identifier'].apply(
    lambda x: item_visibility_mean.get(x, overall_visibility_mean)
)
print(f"Zero 'Item_Visibility' imputed. Remaining zeros: {(combined_df['Item_Visibility'] == 0).sum()}")

# Create 'Item_Type_Combined'
combined_df['Item_Type_Combined'] = combined_df['Item_Identifier'].str[:2].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
print("Created 'Item_Type_Combined' feature.")

# Create 'Outlet_Age' (using 2013 as reference year for data consistency)
combined_df['Outlet_Age'] = 2013 - combined_df['Outlet_Establishment_Year']
print("Created 'Outlet_Age' feature.")

# Correct 'Item_Fat_Content' for Non-Consumable
combined_df.loc[combined_df['Item_Type_Combined'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
print("Corrected 'Item_Fat_Content' for Non-Consumable items.")

# Drop redundant columns
combined_df.drop(['Outlet_Establishment_Year'], axis=1, inplace=True)
print("Dropped redundant columns.")

# Check for any remaining NaNs
print(f"Remaining NaNs in combined_df: \n{combined_df.isnull().sum()}")

# --- 4. Prepare Data for AutoGluon ---
train_processed = combined_df.iloc[:train_len].copy()
test_processed = combined_df.iloc[train_len:].drop(columns=['Item_Outlet_Sales']).copy()  # Drop target here
target = 'Item_Outlet_Sales'
predictor_path = 'AutogluonModels/'

# --- 5. Model Building with AutoGluon ---
print("\n--- Starting AutoGluon Training ---")
try:
    predictor = TabularPredictor(label=target, path=predictor_path, eval_metric='root_mean_squared_error').fit(train_processed)
    print("AutoGluon training complete.")
except Exception as e:
    print(f"Error during AutoGluon training: {e}")
    exit()

# --- 6. Evaluation ---
print("\n--- Evaluating AutoGluon Model ---")
leaderboard = predictor.leaderboard(silent=True)
print("\n--- AutoGluon Model Leaderboard ---")
print(leaderboard)

# --- 7. Prediction and Submission ---
print("\n--- Generating Predictions and Submission File ---")
try:
    # Predict on test data
    test_predictions = predictor.predict(test_processed)
    print("Predictions generated successfully. Sample predictions:", test_predictions.head())

    # Create submission
    submission = pd.DataFrame({
        'Item_Identifier': test_df['Item_Identifier'],
        'Outlet_Identifier': test_df['Outlet_Identifier'],
        'Item_Outlet_Sales': test_predictions.values  # Use .values to avoid index mismatch
    })

    # Ensure no negative sales
    submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].clip(lower=0)

    # Save submission
    submission.to_csv('submission_autogluon.csv', index=False)
    print("Submission file 'submission_autogluon.csv' created successfully.")
    print(submission.head())  # Debug: Print first few rows of submission
except Exception as e:
    print(f"Error during prediction or submission creation: {e}")