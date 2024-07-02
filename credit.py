import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the credit card fraud dataset
data = pd.read_csv('creditcard.csv')

# Separate the features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Handle NaN values in the target variable
y = y.fillna(0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Function to classify a credit card transaction
def classify_transaction():
    print("Enter the credit card transaction details:")

    # Get user input for the transaction features
    amount = float(input("Transaction Amount: "))
    time = int(input("Time: "))
    v1 = float(input("V1: "))
    v2 = float(input("V2: "))
    v3 = float(input("V3: "))
    v4 = float(input("V4: "))
    v5 = float(input("V5: "))
    v6 = float(input("V6: "))
    v7 = float(input("V7: "))
    v8 = float(input("V8: "))
    v9 = float(input("V9: "))
    v10 = float(input("V10: "))
    v11 = float(input("V11: "))
    v12 = float(input("V12: "))
    v13 = float(input("V13: "))
    v14 = float(input("V14: "))
    v15 = float(input("V15: "))
    v16 = float(input("V16: "))
    v17 = float(input("V17: "))
    v18 = float(input("V18: "))
    v19 = float(input("V19: "))
    v20 = float(input("V20: "))
    v21 = float(input("V21: "))
    v22 = float(input("V22: "))
    v23 = float(input("V23: "))
    v24 = float(input("V24: "))
    v25 = float(input("V25: "))
    v26 = float(input("V26: "))
    v27 = float(input("V27: "))
    v28 = float(input("V28: "))

    # Create a DataFrame with the user input
    transaction = pd.DataFrame({
        'Time': [time],
        'V1': [v1],
        'V2': [v2],
        'V3': [v3],
        'V4': [v4],
        'V5': [v5],
        'V6': [v6],
        'V7': [v7],
        'V8': [v8],
        'V9': [v9],
        'V10': [v10],
        'V11': [v11],
        'V12': [v12],
        'V13': [v13],
        'V14': [v14],
        'V15': [v15],
        'V16': [v16],
        'V17': [v17],
        'V18': [v18],
        'V19': [v19],
        'V20': [v20],
        'V21': [v21],
        'V22': [v22],
        'V23': [v23],
        'V24': [v24],
        'V25': [v25],
        'V26': [v26],
        'V27': [v27],
        'V28': [v28],
        'Amount': [amount]
    })

    # Make a prediction using the trained model
    prediction = model.predict(transaction)

    # Print the prediction result
    if prediction[0] == 0:
        print("The transaction is not fraudulent.")
    else:
        print("The transaction is fraudulent.")

# Call the classify_transaction function to get user input and make a prediction
classify_transaction()
