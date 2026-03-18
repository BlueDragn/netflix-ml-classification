'''
V2- Feature engineered ML pipeline
Focus: improving model input representation

'''

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Step 2: Load the dataset
df = pd.read_csv("data/processes/cleaned_data.csv")
print("Data loaded successfully.")

#General dataset exploration
print(df.head(10)) # Display the first 10 rows of the dataset to understand its structure and content.
print("\nShape",df.shape) # Check the shape of the dataset to understand how many samples and features it contains.
print("\nColumns: ", df.columns) # List the column names to identify the features and target variable.
df.info()# Get a summary of the dataset, including data types and non-null counts.
print(df.dtypes) # Check the data types of each column to identify which features are numerical and which are categorical.

#summary statistics
print(df.describe()) # Get statistical summaries of the numerical features to understand their distributions and identify any potential outliers.
print(df.describe(include = "all")) # Get summaries of all columns, including categorical ones

#Data quality checks
print("\nSum_of_Null",df.isnull().sum()) # Check for missing values in each column to identify if any data cleaning is needed before feature engineering.
print("\nDuplicate rows: ", df.duplicated().sum()) # Check for duplicate rows in the dataset to ensure data quality before proceeding with feature engineering.


# ===================================
# Feature Engineering
# ===================================

# Duration -> duration_number + duration_type
# Example: "90 min" -> duration_number: 90, duration_type: "min"
# Example: "1 Season" -> duration_number: 1, duration_type: "Season

df["duration_number"] = df["duration"].str.extract(r"(\d+)") # Extract the numeric part of the duration and convert it to an integer.
df["duration_number"] = df["duration_number"].astype(float) # Convert the extracted numeric part to a float data type for numerical analysis.
df["duration_type"] = df["duration"].str.replace(r"\d+", "", regex=True).str.strip() # Extract the non-numeric part of the duration and remove any leading/trailing whitespace.

# listed_in -> listed_in_genres (list of genres)
# Example: "Action, Comedy" -> listed_in_genres: ["Action", "Comedy"]
df["genre_count"] = df["listed_in"].apply(lambda x: len(x.split(", "))) # Count the number of genres listed in the "listed_in" column to create a new feature representing genre diversity.

#Verify new features
print("\nFeature Engineering Checks:")
print(df[["duration", "duration_number", "duration_type", "genre_count"]].head())


#Define target variable and features
y = df["type"] # Define the target variable 'y' as the 'release_year'
X = df[[ "release_year","genre_count","rating","duration_number","duration_type"]] # Define the feature set 'X' with the original and newly engineered features.

print("\nTarget (y) sample:")
print(y.head()) # Display the first few samples of the target variable to verify its correctness.
print("\nFeatures (X) sample:")
print(X.head()) # Display the first few samples of the feature set to verify the new features


# ===================================
# Preprocessing Pipeline
# ===================================

#feature preparation
#Identify categorical and numerical columns
numeric_features = ["release_year","genre_count","duration_number"] # Define the list of numerical features to be processed in the pipeline.
categorical_features = ["duration_type","rating"] # Define the list of categorical features to be processed in the pipeline.

#Numerical pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")) # Impute missing values in numerical features using the median strategy to handle any potential missing data.
])

#Categorical pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # Impute missing values in categorical features using the most frequent strategy to handle any potential missing data.
    ("encoder", OneHotEncoder(handle_unknown="ignore")) # Encode categorical features using One-Hot Encoding to convert them into a format suitable for machine learning models, while ignoring any unknown categories during transformation.
])

#combine pipelines into a preprocessor
#create the preprocessor pipeline
preprocessor = ColumnTransformer( transformers=[
("cat", categorical_pipeline, categorical_features), # Apply the categorical pipeline to the specified categorical features.
    ("num",numeric_pipeline, numeric_features)
]
)

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the dataset into training and testing sets to evaluate the model's performance on unseen data.
preprocessor.fit(X_train) # Fit the preprocessor on the training data to learn the necessary transformations for both categorical and numerical features.

#Transform features using fitted preprocessing pipeline
X_train_processed = preprocessor.transform(X_train) # Apply the fitted preprocessor to the training features to create the processed feature matrix for model training.
X_test_processed = preprocessor.transform(X_test) # Apply the same preprocessor to the testing features to ensure that the test data is transformed in the same way as the training data for accurate


print("\nEncoded feature matrix shape:", X_train_processed.shape)
print("\nEncoded test feature matrix shape:", X_test_processed.shape)

#Model Training
#Initiate model
model = LogisticRegression(max_iter=1000, class_weight="balanced") # Initialize the Logistic Regression model with a maximum

#Train the model
model.fit(X_train_processed, y_train) # Train the Logistic Regression model using the processed training features and the target variable.
print("\nModel training completed.")

#prediction and evaluation
y_pred = model.predict(X_test_processed) # Use the trained model to make predictions on the processed test features to evaluate its performance on unseen data.

print("\nModel Evaluation:" )

#Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred)) # Calculate and print the accuracy of the model by comparing the true labels with the predicted labels.

#Confusing Matrix
print("\nConfusing Matrix:\n", confusion_matrix(y_test, y_pred)) # Generate and print the confusion matrix to visualize the performance of the classification model in terms of true positives, true negatives, false positives, and false negatives.

#Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred)) # Generate and print a classification report that includes precision, recall, f1-score, and support for each class to provide a detailed evaluation of the model's performance.
