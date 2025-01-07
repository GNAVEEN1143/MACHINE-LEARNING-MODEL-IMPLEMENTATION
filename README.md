# MACHINE-LEARNING-MODEL-IMPLEMENTATION

Name: GADAPA NAVEEN KUMAR

Company: CODTECH IT SOLUTIONS

ID: CT0806AW

Domain: Python Programming

Duration: December 2024 to January 2025

Mentor: N.Santosh


Project: MACHINE-LEARNING-MODEL-IMPLEMENTATION

Overview:

This Python script performs spam email classification using a **Naive Bayes** classifier. Here's an overview of each step in the code:

### 1. **Imports:**
   - **`pandas`**: For data manipulation and reading CSV files.
   - **`train_test_split`** from `sklearn.model_selection`: To split the dataset into training and testing sets.
   - **`TfidfVectorizer`** from `sklearn.feature_extraction.text`: To convert text data into numerical features using the **TF-IDF** (Term Frequency-Inverse Document Frequency) method.
   - **`MultinomialNB`** from `sklearn.naive_bayes`: The Naive Bayes classifier, specifically designed for classification with discrete features, like word counts in text data.
   - **`accuracy_score`, `classification_report`, and `confusion_matrix`** from `sklearn.metrics`: For evaluating the performance of the model.

### 2. **Loading the Dataset**:
   - **`pd.read_csv("C:\\Users\\navee\\Downloads\\spam_or_not_spam.csv.zip", encoding='latin-1')`**: 
     - Loads the dataset from a CSV file (which is in a ZIP format).
     - The `encoding='latin-1'` handles non-UTF-8 characters, which is often required when dealing with certain datasets.
   - **`data.columns`**: Prints the column names in the dataset to verify its structure.

### 3. **Data Preprocessing**:
   - **Column Selection**:
     - The script keeps only the columns `'email'` (the email text) and `'label'` (the classification: spam or ham).
   - **Remove Missing Values**:
     - **`data.dropna()`**: Removes any rows that contain missing values to avoid errors during model training.
   - **Feature and Target Separation**:
     - **`X = data['email']`**: The feature is the email content.
     - **`y = data['label']`**: The target is the label (spam or ham).
   
### 4. **Text Vectorization using TF-IDF**:
   - **`TfidfVectorizer(stop_words='english')`**: 
     - This converts the text data into numerical features.
     - The `stop_words='english'` parameter removes common English words (like "the", "and", etc.) that do not contribute much to classification.
   - **`X_tfidf = vectorizer.fit_transform(X)`**: 
     - The vectorizer is fitted to the email data and transforms it into a sparse matrix of TF-IDF features.

### 5. **Train-Test Split**:
   - **`train_test_split(X_tfidf, y, test_size=0.3, random_state=42)`**:
     - The dataset is split into a training set (70%) and a test set (30%).
     - `random_state=42` ensures reproducibility of the split.

### 6. **Train the Model using Naive Bayes**:
   - **`model = MultinomialNB()`**: 
     - Creates an instance of the Naive Bayes classifier, which is well-suited for text classification tasks.
   - **`model.fit(X_train, y_train)`**: 
     - Trains the Naive Bayes model on the training data.

### 7. **Make Predictions**:
   - **`y_pred = model.predict(X_test)`**: 
     - Makes predictions on the test data.

### 8. **Evaluate the Model**:
   - **Accuracy**: 
     - **`accuracy_score(y_test, y_pred)`**: Computes the accuracy of the model (i.e., the percentage of correct predictions).
     - **`print(f"Accuracy: {accuracy:.4f}")`**: Displays the accuracy to four decimal places.
   - **Classification Report**:
     - **`classification_report(y_test, y_pred)`**: Generates a detailed classification report, which includes precision, recall, F1-score, and support for both classes (spam and ham).
   - **Confusion Matrix**:
     - **`confusion_matrix(y_test, y_pred)`**: Displays the confusion matrix, which shows the true positive, true negative, false positive, and false negative counts, helping to understand how well the model is performing for each class.

### Summary of Evaluation Metrics:
- **Accuracy**: The proportion of correct predictions.
- **Precision**: How many of the predicted spam emails were actually spam (for spam class) or ham (for ham class).
- **Recall**: How many of the actual spam/ham emails were correctly identified by the model.
- **F1-score**: The harmonic mean of precision and recall, giving a balanced evaluation.
- **Confusion Matrix**: Shows the distribution of true and false positives and negatives, offering more insights into the model’s performance.

### Key Observations:
- **Naive Bayes Classifier** is appropriate for text classification because it works well with word frequencies, which is what the TF-IDF vectorizer extracts from the email data.
- **TF-IDF Vectorization** helps in representing the email text in a way that captures important word features while ignoring common, less informative words (like "the", "is", etc.).
- The script assumes that the dataset has the `email` and `label` columns correctly named. If the dataset structure differs, column names may need to be adjusted.

### Conclusion:
This script implements a spam classification system that reads an email dataset, vectorizes the email content using TF-IDF, trains a Naive Bayes classifier, and evaluates the model using accuracy, a classification report, and a confusion matrix. The workflow is well-suited for tasks like spam detection in email filtering systems.

OUTPUT:

![Screenshot 2025-01-07 163539](https://github.com/user-attachments/assets/3d467a95-f7f6-47d7-83d0-69c94d3b97b7)
