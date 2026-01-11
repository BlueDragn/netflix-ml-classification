import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


#Step1
#Load cleaned data from project 1
df = pd.read_csv("data/processes/cleaned_data.csv")
print("Data loaded successfully.")
print("shape:", df.shape)
print(df.head())

# Define target variable and features
y = df["type"]

print("\nTarget (y) sample:")
print(y.head())

#Feature selection
X = df[["release_year", "rating", "duration", "listed_in"]]
print("\nFeatures (X) sample:")
print(X.head())

#sanity Checks
print("\nX shape:", X.shape)
print("y shape",y.shape)

assert X.shape[0] == y.shape[0], "Number of samples in X and y do not match."

#Step 2 feature preparation
#Identify categorical and numerical columns
numeric_features = ["release_year"]
categorical_features = ["rating", "duration", "listed_in"]

#create the preprocessor pipeline
preprocessor = ColumnTransformer( transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
]
)
#Apply encoding to features
X_processed = preprocessor.fit_transform(X)
print("\nEncoded feature matrix shape:", X_processed.shape)

#step 3 Train/Test Split

#Separate data used for learning from data used for evaluation: data separation
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

#Step 4 Model Training

#Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)
#Train the model
model.fit(X_train, y_train)
print("\nModel training completed.")

#Step 5 Model Evaluation: Evaluate the trained model on unseen data and understand its performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))