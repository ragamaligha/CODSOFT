import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    data = pd.read_csv('spam.csv', encoding='latin-1')
except UnicodeDecodeError:
    print("Error: Unable to decode the dataset. Please check the file encoding.")
    exit()

# Preprocess the data
X = data['message']
y = data['label']

# Convert the text data to numerical features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to classify a given SMS message
def classify_sms(message):
    try:
        message_vectorized = vectorizer.transform([message])
        prediction = model.predict(message_vectorized)[0]
        return prediction
    except UnicodeDecodeError:
        print(f"Error: Unable to decode the message '{message}'.")
        return None

# Get user input and classify the message
user_message = input("Enter a SMS message to classify: ")
prediction = classify_sms(user_message)
if prediction is not None:
    print(f"The message '{user_message}' is classified as '{prediction}'.")