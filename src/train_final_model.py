import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main():
    # =============================
    # 1. Load Data
    # =============================
    DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")
    df = pd.read_csv(DATA_PATH)

    print("\n=== Data loaded successfully ===")

    # =============================
    # 2. Define Features & Target
    # =============================
    X = df[["rating", "release_year"]]
    y = df["type"]

    # =============================
    # 3. Train-Test Split
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =============================
    # 4. Preprocessing Pipeline
    # =============================
    numeric_features = ["release_year"]
    categorical_features = ["rating"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # =============================
    # 5. Fit & Transform
    # =============================
    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # =============================
    # 6. Train Model
    # =============================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_transformed, y_train)

    # =============================
    # 7. Evaluate Model
    # =============================
    y_pred = model.predict(X_test_transformed)

    print("\n=== Model Evaluation===")
    print("\n Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print("\n Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

    # =============================
    # 8. Save Artifacts
    # =============================
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print("\n=== Model and preprocessor saved successfully! ===")


# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    main()
