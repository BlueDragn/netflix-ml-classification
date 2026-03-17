<!--
ENGINEERING LOG TEMPLATE

Date:
Topic:

What I did:
Describe the step implemented

Why:
Explain the reasoning behind the changes.

Observation:
What happened after running the code?

Insight:
What concept become clearer?

Next Step:
What will be done next?

-->
# Engineering Log

---

## 14-03-2026
**Topic :**  
- Start of Project 2 revision

**What I did**
- Started project revision and reviewed the existing ML pipeline.

**why**
- To move from guided project to a fully understood implementation.

**Observation**
- The current pipeline includes:  
  1. dataset inspection,  
  2. feature/target separation,  
  3. preprocessing with ColumnTransformer,  
  4. train/test split,  
  5. Logistical Regression training,  
  6. and evaluation

  **Insight**
  - The `duration` column strongly correlates with target variable because movies are measured in minutes and TV shows in seasons.  
  This may create a shortcut for the model.

  **Next Step**
  - Analyze features further and design an experiment to test model dependency on the duration features.

  ---

## 15-03-2026

**Topic**
- Revise the ML pipeline

**What I did**
- Inspected the dataset, discussed why it is important to verify data before transforming.
-  discussed the feature  and final feature set, dived deep into feature engineering.
- Introduced feature engineering ideas (`duration_number`, `duration_type`, `genre_count`)
- discussed concepts like overfiting,underfiting,generalization.

Restructured the projects:
- Renamed `train_model.py` -> `train_movdel_v1.py` (baseline pipeline).
- Created `train_model_v2.py` for improvements.
- Created `docs/` folder with `engineering_log.md` and `engineering.notes.md`

**Why**
- To Preserve the original guided implementation as a baseline while creating a separate version for improvements and experiment.

**Observation**  
- The existing pipeline already demonstrates key ML workflow components including dataset inspection, preprocessing with `ColumnTransformer`, categorical encoding using `OneHotEncoder`, train/test split, Logistic Regression training, and evaluation metrics.

**Concepts Discussed**  
- Supervised learning, classification, target variable definition, feature selection vs feature engineering, dataset shortcuts, train/test split, generalization, and overfitting.

**Next Step**  
- Start development of `train_model_v2.py` and implement feature engineering (`duration_number`, `duration_type`, `genre_count`).  
Plan Experiment 1 to compare model performance with and without the duration feature.

---

## 2026-03-16

**Topic**  
Feature Engineering and Pipeline Understanding

**What I did**  
**These were done using paper and pencil.**

Added feature engineering in V2:
- Created `duration_number` and `duration_type` from `duration`
- Created `genre_count` from `listed_in`

Understood how pandas functions work (`.str`, `apply`, `lambda`).

Compared V1 and V2 pipelines to see the difference between raw features and engineered features.

Reviewed and corrected the ML workflow.



**Why**  
To better understand how data is transformed before training the model and to make sure the pipeline is correct.

**Observation**  
Raw text features can be converted into simple numeric features which are easier for the model to learn from.

**Concepts Discussed**  
Feature engineering, feature selection, encoding, pandas basics, preprocessing pipeline, train/test split, and data leakage.

**Insight**  
Preprocessing should happen after splitting the data using a pipeline, otherwise it can cause data leakage.

**Next Step**  
Implement the preprocessing pipeline in V2 and run the full pipeline (train → predict → evaluate).

## 📅 Date: March 17

## 🎯 Objective
Improve V1 model by building a correct ML preprocessing pipeline and fixing training issues.

---

## ✅ Work Completed

### 1. Feature Engineering (V2 Foundation)
- Created new features:
  - `duration_number` → numeric value from duration
  - `duration_type` → unit (min / Season)
  - `genre_count` → number of genres
- Verified new features using sample outputs

---

### 2. Identified Problem During Model Training
- Error: `ValueError: Input X contains NaN`
- Root cause:
  - Missing values were not handled
  - LogisticRegression cannot accept NaN

---

### 3. Built Proper Preprocessing Pipeline
- Separated features:
  - Numerical → `release_year`, `genre_count`, `duration_number`
  - Categorical → `rating`, `duration_type`

- Created pipelines:
  - Numerical pipeline:
    - Median imputation
  - Categorical pipeline:
    - Most frequent imputation
    - One-hot encoding

- Combined using `ColumnTransformer`

---

### 4. Fixed Data Leakage Issue (Major Improvement)
- Changed pipeline order:

❌ Old (V1):
- fit_transform on full dataset → then split

✅ New (V2):
- split → fit on train → transform train & test

---

### 5. Model Training (Working Pipeline)
- Used LogisticRegression
- Successfully trained model using processed data
- No NaN errors

---

### 6. Model Evaluation Added
- Metrics implemented:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

### 7. Key Observation (Critical Insight)
- Achieved **100% accuracy**
- Confusion matrix shows zero errors

📌 Interpretation:
- Model is using `duration` as a perfect separator:
  - Movies → minutes
  - TV Shows → seasons

⚠️ This indicates:
- Feature dominance / shortcut learning
- Not true generalization

---

## 🧠 Key Learnings

- Importance of handling missing values (imputation)
- Difference between `fit`, `transform`, and `fit_transform`
- Why preprocessing must be done after train/test split
- How data leakage happens and how to prevent it
- How a feature can dominate model behavior
- Why perfect accuracy can be misleading

---

## 🚀 Next Step

- Run Experiment 1:
  - Train model **without duration features**
  - Compare performance with and without duration