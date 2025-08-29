# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
# This section loads the training, test, and sample submission files.
# It's good practice to wrap this in a try-except block to handle file not found errors.
print("--- Loading Data ---")
try:
    train_df = pd.read_csv('train_v9rqX0R.csv')
    test_df = pd.read_csv('test_AbJTz2l.csv')
    sample_submission_df = pd.read_csv('sample_submission_8RXa3c6.csv')
    print("Data loaded successfully.")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please make sure 'train_v9rqX0R.csv', 'test_AbJTz2l.csv', and 'sample_submission_8RXa3c6.csv' are in the same directory.")
    exit()

# For easier data manipulation, we combine the train and test sets.
# We'll split them back up before modeling.
train_len = len(train_df)
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print("Combined shape:", combined_df.shape)


# --- 2. Data Cleaning & Imputation ---
print("\n--- Starting Data Cleaning & Imputation ---")

# Impute missing 'Item_Weight' with the mean weight of the same item.
# This is a logical approach as the same item should have a similar weight.
item_avg_weight = combined_df.pivot_table(values='Item_Weight', index='Item_Identifier')
missing_weight_mask = combined_df['Item_Weight'].isnull()
combined_df.loc[missing_weight_mask, 'Item_Weight'] = combined_df.loc[missing_weight_mask, 'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x, 'Item_Weight'])
print("Missing 'Item_Weight' values imputed.")

# Impute missing 'Outlet_Size' with the mode based on 'Outlet_Type'.
# This assumes that outlets of the same type are likely to be of a similar size.
outlet_size_mode = combined_df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
missing_size_mask = combined_df['Outlet_Size'].isnull()
combined_df.loc[missing_size_mask, 'Outlet_Size'] = combined_df.loc[missing_size_mask, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print("Missing 'Outlet_Size' values imputed.")

# Standardize the 'Item_Fat_Content' categories.
combined_df['Item_Fat_Content'] = combined_df['Item_Fat_Content'].replace({'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'})
print("'Item_Fat_Content' values standardized.")


# --- 3. Feature Engineering ---
print("\n--- Starting Feature Engineering ---")

# Impute 'Item_Visibility' where it is 0, as a product cannot have zero visibility.
# We use the mean visibility for that specific item.
item_visibility_mean = combined_df.pivot_table(values='Item_Visibility', index='Item_Identifier')
zero_visibility_mask = (combined_df['Item_Visibility'] == 0)
combined_df.loc[zero_visibility_mask, 'Item_Visibility'] = combined_df.loc[zero_visibility_mask, 'Item_Identifier'].apply(lambda x: item_visibility_mean.loc[x, 'Item_Visibility'])
print("Zero 'Item_Visibility' values imputed.")

# Create a broader category for 'Item_Type' from the 'Item_Identifier'.
# This helps the model generalize better.
combined_df['Item_Type_Combined'] = combined_df['Item_Identifier'].apply(lambda x: x[0:2])
combined_df['Item_Type_Combined'] = combined_df['Item_Type_Combined'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
print("Created 'Item_Type_Combined' feature.")

# Create an 'Outlet_Age' feature, which is more intuitive for models than the establishment year.
combined_df['Outlet_Age'] = 2013 - combined_df['Outlet_Establishment_Year']
print("Created 'Outlet_Age' feature.")

# Correct a logical inconsistency: Non-consumable items shouldn't have a fat content.
combined_df.loc[combined_df['Item_Type_Combined']=="Non-Consumable", "Item_Fat_Content"] = "Non-Edible"
print("Corrected 'Item_Fat_Content' for Non-Consumable items.")


# --- 4. Encoding Categorical Variables ---
print("\n--- Encoding Categorical Variables ---")

# Use LabelEncoder for features with a clear order (ordinal).
le = LabelEncoder()
ordinal_features = ['Outlet_Size', 'Outlet_Location_Type', 'Item_Fat_Content', 'Item_Type_Combined']
for col in ordinal_features:
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
print("Label Encoded:", ordinal_features)

# Use One-Hot Encoding for features with no inherent order (nominal).
nominal_features = ['Item_Type', 'Outlet_Type', 'Outlet_Identifier']
combined_df = pd.get_dummies(combined_df, columns=nominal_features)
print("One-Hot Encoded:", nominal_features)

# Drop original columns that are now redundant.
combined_df.drop(['Outlet_Establishment_Year'], axis=1, inplace=True)
print("Dropped original columns.")


# --- 5. Model Building ---
print("\n--- Splitting Data and Building Model ---")

# Split the combined dataframe back into training and testing sets.
train_processed = combined_df[:train_len]
test_processed = combined_df[train_len:]

# The test set doesn't have the target variable, so we drop the empty column.
test_processed.drop(['Item_Outlet_Sales'], axis=1, inplace=True)

# Define the target variable and the feature set for the model.
target = 'Item_Outlet_Sales'
ID_cols = ['Item_Identifier']
features = [col for col in train_processed.columns if col not in [target] + ID_cols]

# Initialize the Gradient Boosting Regressor model with some baseline hyperparameters.
gbr = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

# Train the model on the processed training data.
print("Training Gradient Boosting Regressor...")
gbr.fit(train_processed[features], train_processed[target])
print("Model training complete.")


# --- 6. Evaluation ---
print("\n--- Evaluating Model ---")
# Check RMSE on the training data to get a baseline performance metric.
train_predictions = gbr.predict(train_processed[features])
train_rmse = np.sqrt(mean_squared_error(train_processed[target], train_predictions))
print(f"RMSE on Training Data: {train_rmse:.4f}")

# Use 5-fold cross-validation for a more robust measure of the model's performance.
print("Performing 5-fold cross-validation...")
cv_scores = cross_val_score(gbr, train_processed[features], train_processed[target], cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"Average Cross-Validation RMSE: {np.mean(cv_rmse):.4f}")


# --- 7. Prediction and Submission ---
print("\n--- Generating Predictions and Submission File ---")

# Predict sales on the processed test data.
test_predictions = gbr.predict(test_processed[features])

# Create the submission file in the required format.
submission = pd.DataFrame({
    'Item_Identifier': test_df['Item_Identifier'],
    'Outlet_Identifier': test_df['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions
})

# Ensure there are no negative sales predictions.
submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: max(0, x))

# Save the final submission file.
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
