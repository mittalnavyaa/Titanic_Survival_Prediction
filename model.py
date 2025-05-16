import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the training data
train_df = pd.read_csv("Data_after_EDA.csv")

# Define target and features
target_column = 'Survived'
X_train = pd.get_dummies(train_df.drop(target_column, axis=1), drop_first=True)
y_train = train_df[target_column]

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Load the test data
test_df = pd.read_csv("test.csv")
passenger_ids = test_df['PassengerId']  # Save for final output

# Apply the same preprocessing as training (get_dummies)
X_test = pd.get_dummies(test_df, drop_first=True)

# Align test data columns to training data (add missing columns, drop extras)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Predict Survived on test data
test_df['Survived'] = model.predict(X_test)

# Save the result
output = test_df[['PassengerId', 'Survived']]
output.to_csv("submission.csv", index=False)

print("✅ Predictions saved to 'submission.csv'")
