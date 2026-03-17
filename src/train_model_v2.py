'''
V2- Feature engineered ML pipeline
Focus: improving model input representation

'''

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Step 2: Load the dataset
df = pd.read_csv("data/processes/cleaned_data.csv")
print("Data loaded successfully.")

#General dataset exploration
print(df.head(10)) # Display the first 10 rows of the dataset to understand its structure and content.
print(df.shape) # Check the shape of the dataset to understand how many samples and features it contains.
print(df.columns) # List the column names to identify the features and target variable.
df.info()# Get a summary of the dataset, including data types and non-null counts.
print(df.dtypes) # Check the data types of each column to identify which features are numerical and which are categorical.

#summary statistics
print(df.describe()) # Get statistical summaries of the numerical features to understand their distributions and identify any potential outliers.
print(df.describe(include = "all")) # Get summaries of all columns, including categorical ones

#Data quality checks
print(df.isnull().sum()) # Check for missing values in each column to identify if any data cleaning is needed before feature engineering.
print("Duplicate rows: ", df.duplicated().sum()) # Check for duplicate rows in the dataset to ensure data quality before proceeding with feature engineering.


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
y = df["type"] # Define the target variable 'y' as the 'type'
X = df[["release_year", "rating", "duration_number", "duration_type", "genre_count"]] # Define the feature set 'X' with the original and newly engineered features.

print("\nTarget (y) sample:")
print(y.head()) # Display the first few samples of the target variable to verify its correctness.
print("\nFeatures (X) sample:")
print(X.head()) # Display the first few samples of the feature set to verify the new features


# ===================================
# Preprocessing Pipeline
# ===================================

#feature preparation
#Identify categorical and numerical columns
numeric_features = ["release_year","genre_count","duration_number"]
categorical_features = ["rating", "duration_type" ]

#create the preprocessor pipeline
preprocessor = ColumnTransformer( transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
]
)

#Apply encoding to features
X_processed = preprocessor.fit_transform(X)
print("\nEncoded feature matrix shape:", X_processed.shape) # Check the shape of the encoded feature matrix to ensure that the preprocessing step has been applied correctly and to understand the dimensionality of the input for model training.