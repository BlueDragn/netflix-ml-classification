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

## Failure Test 2: Unseen Category Handling

### Objective
Test whether the model can handle unseen categorical values in input data without failure.

---

### Method

Injected an unseen category into the test dataset after train-test split:

    # Inject unseen category into test set
    X_test.loc[X_test.index[0], "rating"] = "NC-17"

(Note: "NC-17" is assumed not present in training data)

---

### Result

- Model executed successfully (no crash)
- No warnings observed
- Accuracy ≈ **0.559**

**Confusion Matrix:**

    [[473 741]
     [ 36 512]]

- Performance remained consistent with baseline

---

### Conclusion

- OneHotEncoder with `handle_unknown="ignore"` successfully handled unseen category
- System remained stable and did not fail
- Unseen category was ignored during encoding (no new feature created)
- No significant impact on performance observed
- Model predictions continue, but unseen categories provide no useful signal

---


## Failure Test 3: Invalid Data Type Injection

### Objective
Test how the pipeline behaves when incorrect data types are introduced into numeric and categorical features.

---

### Method
Injected invalid values into the test set after train-test split:

- Set `release_year` (numeric) to a string value
- Set `rating` (categorical) to an out-of-range numeric value


```
X_test.loc[X_test.index[0], "release_year"] = "Two Thousand Twenty"  
X_test.loc[X_test.index[1], "rating"] = 999  
```

---

### Result

-  Pipeline failed during preprocessing  

- Error:  
1. ValueError: Cannot use median strategy with non-numeric data  
2. could not convert string to float: 'Two Thousand Twenty'  

---

### Conclusion

- Pipeline is not robust to incorrect data types  
- Numeric pipeline fails when non-numeric values are introduced  
- Categorical pipeline handled unexpected value safely  
- System requires input validation before preprocessing 


---



## Failure Test 4: Extreme Values / Outliers

### Objective
Test how the pipeline behaves when extreme numerical values (outliers) are introduced into the input data.

---

### Method
Injected extreme values into the numeric feature `release_year` in the test set after train-test split:

- Set one value far in the future (3000)
- Set one value far in the past (1800)

```

X_test.loc[X_test.index[0], "release_year"] = 3000  
X_test.loc[X_test.index[1], "release_year"] = 1800  
```
---

### Result

✔ Pipeline executed successfully (no crash)

Accuracy: ~0.559  

Confusion Matrix:
[[473 741]  
 [ 36 512]]

Classification behavior remained similar to previous runs

---

### Conclusion

- Pipeline is numerically stable under extreme values  
- StandardScaler handled out-of-range values without failure  
- Model predictions were not significantly affected by small number of outliers  
- However, extreme values do not improve model performance and may introduce instability if frequent  

This shows the system is stable but not necessarily robust to unrealistic data distributions 

---
