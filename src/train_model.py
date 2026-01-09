import pandas as pd

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