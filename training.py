import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your collected landmark data CSV
data = pd.read_csv('sign_data.csv', header=None)

# Last column is the label
X = data.iloc[:, :-1]  # landmarks features
y = data.iloc[:, -1]   # gesture labels

# Split data to train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Training accuracy:", model.score(X_test, y_test))

# Save the model
joblib.dump(model, 'sign_model.pkl')
