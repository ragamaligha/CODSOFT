import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the customer churn dataset
data = pd.read_csv('Churn_Modelling.csv')

# Preprocess the data
X = data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = data['Exited']

# Encode the categorical features
geography_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
X['Geography'] = geography_encoder.fit_transform(X['Geography'])
X['Gender'] = gender_encoder.fit_transform(X['Gender'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost classifier
model = XGBClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Get user input for new customer data
credit_score = int(input("Enter credit score: "))
geography = input("Enter geography (Germany, France, or Spain): ")
gender = input("Enter gender (Male or Female): ")
age = int(input("Enter age: "))
tenure = int(input("Enter tenure: "))
balance = float(input("Enter balance: "))
num_products = int(input("Enter number of products: "))
has_cr_card = int(input("Has credit card (0 or 1): "))
is_active_member = int(input("Is active member (0 or 1): "))
estimated_salary = float(input("Enter estimated salary: "))

# Encode the new customer data
new_customer = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Handle new labels for Geography and Gender
if geography not in geography_encoder.classes_:
    geography_encoder.classes_ = np.append(geography_encoder.classes_, geography)
new_customer['Geography'] = geography_encoder.transform(new_customer['Geography'])

if gender not in gender_encoder.classes_:
    gender_encoder.classes_ = np.append(gender_encoder.classes_, gender)
new_customer['Gender'] = gender_encoder.transform(new_customer['Gender'])

# Scale the new customer data
new_customer_scaled = scaler.transform(new_customer)

# Predict churn probability
churn_probability = model.predict_proba(new_customer_scaled)[0][1]

print(f"Probability of churn: {churn_probability:.2f}")
