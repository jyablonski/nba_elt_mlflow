import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from src.utils import sql_connection

# one hot encoding is not necessary for `is_top_players`
# 0 < 1 < 2 - it actually has ordinal value
# if you asked ppl what their fave color was and used 1 for red, 2 for blue, 3 for green
# etc then that would need to be one hot encoded because the differential in value
# has no meaningful significance.

# conn = sql_connection('ml_models')
# past_games = pd.read_sql_query('select * from ml_past_games_odds_analysis;', conn).query(f"outcome.notna()")
past_games = pd.read_csv("past_games_2023-10-18.csv")
past_games["outcome"] = past_games["outcome"].replace({"W": 1, "L": 0})

past_games_outcome = past_games["outcome"]

past_games_ml_dataset = past_games.drop(
    [
        "home_team_predicted_win_pct",
        "away_team_predicted_win_pct",
        "ml_accuracy",
        "ml_money_col",
        "home_implied_probability",
        "away_implied_probability",
        "outcome",
        "ml_prediction",
        "actual_outcome",
        "proper_date",
        "away_team",
        "home_team",
    ],
    axis=1,
)

## my old way
past_games_outcome = past_games_outcome.to_numpy()
training_set = past_games_ml_dataset.to_numpy()

clf_linear_svc = LinearSVC(random_state=0).fit(training_set, past_games_outcome)
clf_svc = SVC(random_state=0).fit(training_set, past_games_outcome)
clf = LogisticRegression(random_state=0).fit(training_set, past_games_outcome)

print(f"Linear SVC score was {clf_linear_svc.score(training_set, past_games_outcome)}")
print(f"SVC score was {clf_svc.score(training_set, past_games_outcome)}")
print(f"Logistic Regression score was {clf.score(training_set, past_games_outcome)}")

############################### START Random Forest ###########################
past_games_ml_dataset = pd.get_dummies(
    past_games_ml_dataset,
    columns=["away_is_top_players", "home_is_top_players"],
    drop_first=True,
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    past_games_ml_dataset, past_games_outcome, test_size=0.2, random_state=42
)


# Use StandardScaler to normalize the numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Predictions
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Extract feature importances
feature_importances = random_forest_model.feature_importances_

# Create a DataFrame to display feature names and their importances
feature_importances_df = pd.DataFrame(
    {"Feature": past_games_ml_dataset.columns, "Importance": feature_importances}
)
feature_importances_df = feature_importances_df.sort_values(
    by="Importance", ascending=False
)

# Print or visualize the feature importances
print(feature_importances_df)

# Visualize the feature importances
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importances_df["Feature"],
    feature_importances_df["Importance"],
    color="skyblue",
)
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.show()

############################### END Random Forest ###########################

############################### START Random Forest RFE ###########################

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    past_games_ml_dataset, past_games_outcome, test_size=0.2, random_state=42
)

# Create a Random Forest model
model = RandomForestClassifier(random_state=42)

# Initialize RFE
rfe = RFE(
    model, n_features_to_select=6
)  # Select the number of features you want to keep

# Fit RFE
fit = rfe.fit(X_train, y_train)

# Print the rankings of features
print("Feature Rankings:")
for feature_rank in zip(past_games_ml_dataset.columns, fit.ranking_):
    print(feature_rank)

# Get the selected features
selected_features = [
    feature
    for feature, rank in zip(past_games_ml_dataset.columns, fit.ranking_)
    if rank == 1
]
print("\nSelected Features:")
print(selected_features)

# Train the model with selected features
model.fit(X_train[selected_features], y_train)

# Predictions
y_pred = model.predict(X_test[selected_features])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy with Selected Features:", accuracy)

############################### END Random Forest RFE ###########################

############################### START SVM ###########################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    past_games_ml_dataset, past_games_outcome, test_size=0.2, random_state=42
)

# Create an SVM model
model = SVC(kernel="linear")  # You can choose different kernels based on your problem

# Initialize RFE
rfe = RFE(
    model, n_features_to_select=10
)  # Select the number of features you want to keep

# Fit RFE
fit = rfe.fit(X_train, y_train)

# Print the rankings of features
print("Feature Rankings:")
for feature_rank in zip(past_games_ml_dataset.columns, fit.ranking_):
    print(feature_rank)

# Get the selected features
selected_features = [
    feature
    for feature, rank in zip(past_games_ml_dataset.columns, fit.ranking_)
    if rank == 1
]
print("\nSelected Features:")
print(selected_features)

# Train the model with selected features
model.fit(X_train[selected_features], y_train)

# Predictions
y_pred = model.predict(X_test[selected_features])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy with Selected Features:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)
############################### END SVM ###########################

############################### START Log Regression Ensemble ###########################
X_train, X_test, y_train, y_test = train_test_split(
    past_games_ml_dataset, past_games_outcome, test_size=0.2, random_state=42
)

# Use StandardScaler to normalize the numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_model1 = LogisticRegression(random_state=42)
logreg_model2 = LogisticRegression(random_state=42)

voting_classifier = VotingClassifier(
    estimators=[
        ("logreg1", logreg_model1),
        ("logreg2", logreg_model2),
    ],  # Add more models if desired
    voting="soft",  # Use 'soft' for probability-based voting
)

voting_classifier.fit(X_train, y_train)

y_pred = voting_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

############################### END Log Regression Ensemble ###########################


############################### START Log Regression Bootstrap #########################
num_bootstrap_samples = 100  # Adjust as needed
logreg_models = []

X_train, X_test, y_train, y_test = train_test_split(
    past_games_ml_dataset, past_games_outcome, test_size=0.2, random_state=42
)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

for i in range(num_bootstrap_samples):
    # Perform bootstrap sampling (sampling with replacement)
    X_bootstrap, y_bootstrap = resample(X_train, y_train, replace=True, random_state=i)

    # Create a binary classification model (e.g., RandomForestClassifier)
    model = LogisticRegression(random_state=42)
    model.fit(X_bootstrap, y_bootstrap)

    # Append the trained model to the list
    logreg_models.append(model)


# Initialize an array to store predictions
all_predictions = np.zeros((num_bootstrap_samples, len(X_test)))

# Predict using each model
for i, model in enumerate(logreg_models):
    predictions_encoded = model.predict(X_test)
    all_predictions[i] = predictions_encoded

# Combine predictions using voting (majority vote)
combined_predictions = np.round(np.mean(all_predictions, axis=0))

# Evaluate the combined predictions
accuracy = accuracy_score(y_test, combined_predictions)
print("Ensemble Logistic Regression with Bootstrap Accuracy:", accuracy)
