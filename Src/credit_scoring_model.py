# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# Load dataset
data = pd.read_csv('../data/Credit Score Classification Dataset.csv')

# Preprocess data
# Encode categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Education', 'Marital Status', 'Home Ownership'], drop_first=True)

# Map 'Credit Score' to numerical values
credit_mapping = {'Low': 0, 'Average': 1, 'High': 2}
data['Credit Score'] = data['Credit Score'].map(credit_mapping)

# Separate features and target variable
X = data.drop('Credit Score', axis=1)
y = data['Credit Score']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')
print("ROC-AUC Score:", roc_auc)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Save the best model and scaler
best_model = grid_search.best_estimator_
joblib.dump(best_model, '../models/credit_scoring_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')