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
- Start development of `train_model_v2.py` and implement feature engineering (`duration_number`, `duration_type`, `genre_count`). Plan Experiment 1 to compare model performance with and without the duration feature.
