'''
V2- Feature engineered ML pipeline
Focus: improving model input representation

'''

# Step 1: Import necessary libraries
import pandas as pd

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