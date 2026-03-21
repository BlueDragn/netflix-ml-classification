# =======================
# 1. Import necessary libraries
# =======================

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =======================
# 2. Load the cleaned dataset from project 1
# =======================
df = pd.read_csv("data/processes/cleaned_data.csv")
print("Data loaded successfully.")

# =======================
# 3. Basic data inspection
# =======================

print("\nshape:", df.shape)
print(df.head())
print(df.columns)
df.info()
#df.isnull().sum()





# =======================
# 5. Local feature engineering
# =======================

# duration column
df["duration_number"] = df["duration"].str.extract(r"(\d+)").astype(float)
df["duration_type"] = df["duration"].str.replace(r"(\d+)", "").str.strip()

df[["duration", "duration_number", "duration_type"]].head()

# listed_in column
df["genre_count"] = df["listed_in"].apply(lambda x: len(x.split(", ")))

df[["listed_in", "genre_count"]].head()


# =======================
# 4. Define X and y
# =======================
X = df[["release_year", "rating", "duration_type", "duration_number", "genre_count"]]
y = df["type"]

# =======================
# 6. Train/Test Split (80/20)
# =======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)


# =======================
# 8. Define preprocessing pipeline (imputation, encoding)
# =======================
numeric_features = ["release_year", "duration_number", "genre_count"]
categorical_features = ["rating", "duration_type"]

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

category_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", category_pipeline, categorical_features),
        ("num",num_pipeline , numeric_features)
    ]
)


# =======================
# 9. Fit pipeline on X_train
# =======================

preprocessor.fit(X_train)
print("\nPreprocessor fitted on training data.")

# =======================
# 10. Transform:
# - X_train-> X_train_processed
# - X_test -> X_test_processed
# =======================

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# =======================
# 11. Train a model on X_train_processed (e.g., Logistic Regression)
# =======================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_processed, y_train)
print("\nModel training completed.")

# =======================
# 12. Evaluate the model on X_test_processed
# =======================
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


