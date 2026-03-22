# 🎬 Netflix Content Classification System  
### Production-Style ML Pipeline with Failure Testing & Robust Preprocessing

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Focus](https://img.shields.io/badge/focus-ML%20Engineering-blueviolet)

---

## 🚀 Why This Project Matters

Most ML projects stop at:
> “Train model → get accuracy”

This project goes further:
- Builds a **reusable ML pipeline**
- Tests **real-world failure scenarios**
- Demonstrates **system-level thinking**

👉 This is closer to how ML systems are built in production.

---

## 🧠 Problem Statement

Classify Netflix content into:
- **Movie**
- **TV Show**

Using structured metadata.

---

## ⚙️ System Architecture

```
Raw Data (Processed)
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ├── Numeric: Impute → Scale
   └── Categorical: Impute → Encode
   ↓
Model (Logistic Regression)
   ↓
Evaluation
   ↓
Saved Artifacts (model + preprocessor)
```

---

## 📊 Features Used (Final Model)

| Feature        | Type        | Reason |
|---------------|------------|--------|
| rating        | Categorical | Strong signal for content type |
| release_year  | Numerical   | Temporal distribution |

> Simpler feature set chosen after experimentation for stability.

---

## 🔧 ML Pipeline

### Preprocessing

**Numerical Pipeline**
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

**Categorical Pipeline**
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

---

### Model
- Logistic Regression (`max_iter=1000`)

---

## 📈 Model Performance

### Accuracy
**~70%**

---

### Classification Report

| Class     | Precision | Recall | F1-score | Support |
|----------|----------|--------|----------|---------|
| Movie    | 0.70     | 0.96   | 0.81     | 1214    |
| TV Show  | 0.56     | 0.11   | 0.18     | 548     |

---

### Confusion Matrix

```
[[1168   46]
 [ 490   58]]
```

---

## 🔍 Key Insights

### ✅ What Works
- Strong detection of **Movies**
- Stable preprocessing pipeline
- Handles:
  - Missing values
  - Unseen categories

### ⚠️ What Breaks / Needs Work
- Poor recall for **TV Shows**
- Class imbalance bias
- Limited feature representation

---

## 🧪 Failure Testing

 This system is tested against **real-world data failures**.

### Tests Performed

| Scenario                     | Outcome |
|----------------------------|--------|
| Missing values (NaN)       | ✅ Handled |
| Unseen categories          | ✅ Handled |
| Wrong data types           | ❌ Failure |
| Extreme values             | ⚠️ Stable |
| Missing columns            | ❌ Failure |

---

### 💡 Engineering Insight

> ML pipelines are robust to **missing data**,  
> but fragile to **schema & type violations**

---

## 🏗️ Project Structure

```
netflix-ml-classification/
│
├── data/
│   └── processed/
│       └── cleaned_data.csv
│
├── docs/
│   ├── engineering_log.md
│   ├── engineering_notes.md
│   └── failure_tests_log.md
|
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── train_model_v1.py
│   ├── train_model_v2.py
│   └── train_final_model.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💾 Saved Artifacts

| File | Description |
|------|------------|
| `model.pkl` | Trained classifier |
| `preprocessor.pkl` | Full preprocessing pipeline |

---

## 🔄 Experimentation Workflow

This project evolved through controlled experiments:

1. **Baseline model**
2. **Feature engineering**
3. **Failure testing (robustness validation)**
4. **Final simplified model**

### Branch Strategy

- `experiments/feature-tests`
- `refactor/baseline-clean`
- `backup/final-experiment-state`
- `main` (final production-ready version)

---

## 📌 Key Takeaways

- Simpler models can be **more stable**
- Feature selection directly impacts **bias**
- Real ML systems must handle:
  - Missing data
  - Unknown categories
  - Schema validation

---

## 🔮 Future Improvements

- Add richer features:
  - duration
  - genre_count
- Handle class imbalance:
  - class weights / SMOTE
- Add input validation layer
- Deploy as API (FastAPI)

---

## 🧑‍💻 Author

**Anshuman**  
AI/ML Engineer (in progress) | Python Backend Developer  

---

## ⭐ Why This Project Stands Out

- Goes beyond modeling → focuses on **ML system design**
- Includes **failure testing (rare in beginner projects)**
- Demonstrates:
  - pipeline engineering
  - debugging mindset
  - real-world thinking

---

## 🔗 Next Steps

- Convert this into a **deployable API**
- Add **monitoring + logging**
- Extend to **multi-feature modeling system**
