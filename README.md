### Predicting Acute Kidney Injury (AKI) in Patients with Acute Myocardial Infarction (AMI)
![AKI Prediction Pipeline](images/AKI_pipeline.jpg)
## 📌 Overview

The main objective of this project is to develop a predictive model for estimating the risk of **Acute Kidney Injury (AKI)** within **7 days** among patients diagnosed with **Acute Myocardial Infarction (AMI)**.

The prediction model uses only clinical and laboratory data collected during the **first 12 hours after ICU admission**. The model is trained and internally validated using the **MIMIC-IV** database, with external validation performed on an independent dataset called **eICU**.
 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kidney-injury-prediction.streamlit.app/)
**Live Web Application:** [kidney-injury-prediction.streamlit.app](https://kidney-injury-prediction.streamlit.app/)

---

## 🛠️ Pipeline

The project is organized into four main stages, corresponding to the Jupyter notebooks in the repository.

### 1. Cohort Selection and AKI Labeling

`create_AMI_AKI_patients.ipynb`

This notebook is responsible for identifying eligible AMI patients and assigning AKI outcome labels.

Main steps include:

* **Cohort extraction:**
  Identify AMI patients using ICD-9 and ICD-10 diagnosis codes.

* **Exclusion criteria:**
  Patients are excluded if they meet any of the following conditions:

  * Pre-existing chronic kidney disease or severe renal disease.
  * Age below 18 years or above 90 years.
  * ICU length of stay shorter than 12 hours.
  * AKI already present before the 12-hour landmark after ICU admission, according to KDIGO criteria.

* **Outcome labeling:**
  Patients are labeled as AKI-positive (`Label = 1`) if they develop **KDIGO stage ≥ 1 AKI** between 12 hours after ICU admission and the following 7 days.

---

### 2. Feature Extraction

`extract_patient_features.ipynb`

This notebook extracts and aggregates patient-level features from the first **12 hours of ICU stay**.

The extracted features include the following categories:

* **Demographics and body measurements**

  * Age
  * Sex
  * Height
  * Weight

* **Vital signs**

  * Heart rate
  * Blood pressure
  * Respiratory rate
  * Temperature
  * Oxygen saturation
  * Other ICU physiological measurements

* **Urine output**

  * Total urine output within the first 12 hours

* **Laboratory tests**

  * Blood chemistry: creatinine, BUN, glucose, etc.
  * Complete blood count: WBC, hemoglobin, platelet count, etc.
  * Cardiac biomarkers: troponin T, CK-MB, etc.

* **Treatments and interventions**

  * Mechanical ventilation
  * Vasopressor use
  * Antibiotic use
  * Aspirin
  * ACE inhibitors
  * Invasive lines and procedures

* **Time-series feature aggregation**

  * Time-dependent variables are transformed into summary statistics such as:

    * Minimum
    * Maximum
    * Mean
    * Median
    * Sum

---

### 3. Data Preprocessing

`preprocess.ipynb`
Located in both `internal validation/` and `external validation/`

This notebook prepares the extracted features for model training and validation.

Main preprocessing steps include:

* **Missing value handling**

  * Features with more than **50% missing values** are removed.
  * Missing categorical values are imputed using `-1` or `0`.
  * Missing numerical values are imputed using `KNNImputer` with `K = 10`.

* **Outlier handling**

  * Capping is applied at the 99th percentile.
  * Winsorization is applied at the 1st and 99th percentiles.

* **Feature scaling**

  * Numerical features are standardized using `StandardScaler`.
  * The scaler is fitted on the training set and then applied to the test or validation set.

---

### 4. Model Training and Evaluation

`training_LR.ipynb`

This notebook trains and evaluates the main prediction model.

#### Feature Selection

Feature selection is performed in three steps:

1. **Univariable Logistic Regression**
   Features with `p-value < 0.05` are retained.

2. **Multicollinearity assessment using VIF**
   Features with acceptable multicollinearity are retained using a threshold of `VIF < 10`.

3. **Multivariable Logistic Regression**
   The final feature set is selected based on multivariable model performance and statistical significance.

#### Model Training

The core predictive model is **Logistic Regression**.

Hyperparameter tuning is performed using `GridSearch`, with the search space including:

* `C`
* `penalty`
* `solver`
* `max_iter`

#### Model Evaluation

Model performance is evaluated using **bootstrapping with 1000 iterations** to estimate **95% confidence intervals**.

The following metrics are reported:

* Accuracy
* Recall / Sensitivity
* Precision / Positive Predictive Value
* F1-score
* ROC-AUC
* AUPRC

#### Model Explainability

Model interpretation is performed using:

* **SHAP values**
* **Odds Ratios (OR)**
This project aims to support early identification of AMI patients at high risk of developing AKI during ICU hospitalization. By using data available within the first 12 hours of ICU admission, the model may help clinicians perform early risk stratification and guide timely preventive interventions.
