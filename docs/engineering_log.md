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

## 11-03-2026
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




