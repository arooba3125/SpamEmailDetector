from preprocess import load_and_preprocess_data, vectorize_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load and preprocess data
df = load_and_preprocess_data("dataset/spam.csv")

# Vectorize text with additional features
X, y, feature_union = vectorize_text(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_classifier.pkl")
joblib.dump(feature_union, "model/feature_union.pkl")  # Save the entire feature union
print("Model training complete.")