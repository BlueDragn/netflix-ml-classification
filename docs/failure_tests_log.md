# Failure Tests Log
---

## Base pipeline (Used Across All Tests)

**Features Used**
- Numeric:  `release_year`
- Categorical:  `rating`

**Preprocessing:**
- Numeric Pipeline:
   - Missing values -> `median` imputation
   - Scaling -> `StandardScaler`
- Categorical Pipeline:
   - Missing values -> `most_frequent` imputation
   - Encoding -> `OneHotEncoder (handle_unknown="ignore")`

**Model:**
- Logistic Regression (with `class_weight = 'balanced'`)

---


## Failure Test 1: Missing Values Handling

### Objective
Test whether the model pipeline can handle missing values (`NaN`) in input data without failure.

---

### Method

We injected missing values into the test dataset after the train-test split to simulate real-world incomplete input data.

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Inject NaN into test data (simulate missing inputs)
    X_test.loc[X_test.index[0], "rating"] = np.nan
    X_test.loc[X_test.index[1], "release_year"] = np.nan

---

### Result

- Model executed successfully (no crash)
- No warnings observed
- Accuracy ≈ **0.558**

**Confusion Matrix:**

    [[472 742]
     [ 36 512]]

- Performance remained consistent with baseline (EXP 6B)

---

### Conclusion

- The preprocessing pipeline successfully handled missing values using imputation
- System is robust to incomplete input data
- Missing values did not cause any failure or instability
- No significant performance degradation observed
- Overall model performance remains limited due to weak feature signal, not missing data issues

---

