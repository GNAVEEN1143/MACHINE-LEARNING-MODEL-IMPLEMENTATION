import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("C:\\Users\\navee\\Downloads\\spam_or_not_spam.csv.zip", encoding='latin-1')

# Check the dataset columns
print(data.columns)

# Update column names according to your dataset
# 'email' is the text and 'label' is the spam/ham classification
data = data[['email', 'label']]  # 'email' for the text, 'label' for the spam/ham classification

# Display the first few rows to verify the dataset
print(data.head())

# Preprocess the data: Remove rows with missing values
data = data.dropna()

# Split the dataset into features (X) and target (y)
X = data['email']  # Feature: email content
y = data['label']  # Target: spam or ham

# Step 4: Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')  # Convert text to numerical features
X_tfidf = vectorizer.fit_transform(X)  # Apply the vectorizer to the email data

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Step 6: Train the Model using Naive Bayes (MultinomialNB)
model = MultinomialNB()  # Naive Bayes classifier for text data
model.fit(X_train, y_train)  # Train the model on the training data

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
