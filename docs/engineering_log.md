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


----
-----







  ## Experiment 1: Remove duration_type

  **Changes**  
  Remove 'duration_type' from feature set.

  **Result:**
  No changes in accuracy(0.9994)  
  Confusing matrix [[1213,1],
                    [0,548]]

  **Insight:**  
  duration_type is redundant; **duration_number** alone provides sufficient signal for **classification**.
  Model does not rely on **duration_type**



## Experiment 2: Only Duration Features

**Changes:**
Used only ['duration_number', 'duration_type'] as input features.

**Result:**
Achieved perfect accuracy (1.0) with zero misclassification:  
[[1214,0],  
   [0,548]]


**Insight**  
Duration features alone are sufficient to completely solve the classification problem.This indicates that the dataset is highly separable based on duration_number, making the problem trivial.Other features so not contribute meaningful additional information.


## Experiment 3: Remove duration_number

**Changes**
Removed 'duration_number' while keeping other features

**Result:**
Accuracy dropped slightly to ~0.9966 with 6 misclassifications:  
 [[1214, 0],  
 [6, 542]]

**Insight**  
Although **duration_number** is a dominant feature, **duration_type** alone still provides strong separability between Movie and TV Show.  
The classification problem remains
nearly trivial even without numeric duration, indicating that duration-related features collectively dominate the dataset.

## Experiment 4: Remove all duration features

**Changes**  
Remove both 'duration_number' and 'duration_type'

**Result:**  
Accuracy dropped significantly to ~0.699 with heavy miscalculation:  
[[1169, 45],  
  [485,63]]

**Insight**  
Without duration features. the model struggles to distinguish between classes.
It becomes heavily biased toward predicting the majority class (Movie),resulting in very poor TV Shows (0.11).This reveals that duration features were critical for separability,and other features provide weak predictive signal.


## Experiment 5: Class Imbalance Handling

**Change**  
Applied class_weight='balanced' in LogisticRegression

**Result:**  
Accuracy decreased to ~0.608 with more balanced predictions:  
[[594, 620],  
[70, 478]]

**Insight**  
Applying class weights significantly improved recall for the minority class (TV Show) from 0.11 to 0.87, but at the cost of reduced performance on the majority class (Movie).  
The model shifted from being biased towards Movies to making more balanced predictions across classes, demonstrating the trade-off between overall accuracy and class-wise fairness.


## Experiment 6A : Remove release_year

**Changes**  
Removed 'release_year' from feature set

**Result**  
Accuracy decreased to ~0.562:  
[[505, 709],  
[63, 485]]

**Insight**  
Removing release_year caused a noticeable drop in performance, indicating that although it is a weak feature, it still contributes useful signal. In the absence of strong features (like duration), even weak features help stabilize model predictions.

## Experiment 6B : Remove genre_count

**Changes**  
Removed 'genre_count' from feature set

**Result:**  
Accuracy decreased to ~0.558:  
[[472, 742],  
  [36, 512]]

**Insight:**  
Removing genre_count caused a slight drop in performance, indicating it contributes weak but useful signal.Similar to release_year, it helps stabilize predictions when strong features are absent. Rating remains the most influential features among the remaining features.


----

## Experiment Summary: Feature Importance & Model Behavior Analysis

A series of controlled experiments were conducted to understand feature importance, model behavior, and dataset characteristics.

### Key Findings:

1. **Dominant Features (Duration)**
   - Using only duration features (`duration_number`, `duration_type`) resulted in near-perfect accuracy (~1.0).
   - These features alone were sufficient to completely separate the classes.
   - This indicates that the dataset is trivially separable due to strong duration-based signals.

2. **Redundant Feature Behavior**
   - Removing `duration_type` had negligible impact.
   - Removing `duration_number` caused only a minor drop in performance.
   - Both features encode similar information, making them partially redundant.

3. **True Dataset Difficulty**
   - Removing all duration features caused accuracy to drop significantly (~0.70).
   - The model became biased toward the majority class (Movie), with very poor recall for TV Shows.
   - This revealed that the problem is not inherently easy; it only appears easy due to dominant features.

4. **Class Imbalance Impact**
   - Applying `class_weight='balanced'` improved recall for the minority class (TV Show) from ~0.11 to ~0.87.
   - However, this reduced performance on the majority class and overall accuracy.
   - Demonstrates the trade-off between accuracy and class-wise fairness.

5. **Weak Feature Analysis**
   - Removing `release_year` and `genre_count` caused small but consistent drops in performance.
   - These features provide weak but useful signals that help stabilize predictions.
   - `rating` emerged as the most influential feature among the remaining features.

---

### Final Feature Hierarchy:

- **Strong:** `duration_number`, `duration_type`
- **Moderate:** `rating`
- **Weak but useful:** `genre_count`, `release_year`

---

### Final Insight:

The model's high performance is primarily driven by dominant duration-based features. When these are removed, the model struggles and becomes biased due to weak remaining signals. This highlights the importance of understanding feature influence and avoiding misleading conclusions from inflated accuracy.


----