# Credit-Risk-Modelling
This project implements a machine learning pipeline to predict the likelihood of loan defaults. By analyzing borrower demographics and loan characteristics, the system identifies high-risk applications to minimize financial loss while maintaining a high quality of service for low-risk customers.

### Model Performance Summary

After evaluating multiple architectures, **XGBoost** was selected for production due to its superior handling of class imbalance and non-linear feature relationships. 

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 83% | 0.60 | 0.69 | 0.64 |
| Random Forest | 91% | 0.86 | 0.71 | 0.78 |
| **XGBoost (Final)** | **92%** | **0.91** | **0.72** | **0.80** |

> **Note:** The final XGBoost model was tuned with a custom decision threshold of **0.20**. This specific configuration was chosen to maximize the detection of defaults (Recall) while maintaining a Precision rate above 90%, ensuring high reliability for the credit department.

### Key Features & Engineering
To improve model "vision," we engineered specific financial ratios that capture the burden of debt more effectively than raw numbers:
* Loan-to-Income Ratio: Normalizes the loan amount against the borrower's earning power.
* Interest-to-Income Ratio: Measures the monthly "drain" on a borrower's paycheck.
* Home Ownership Encoding: Captured as a primary risk indicator (Renters vs. Homeowners).

### Evaluation Metrics: 
We moved beyond simple accuracy to ensure the model performs in "real-world" imbalanced conditions:
* ROC & AUC
The XGBoost model achieved an AUC of 0.926, demonstrating excellent separability between defaulters and non-defaulters across all thresholds.

* Precision-Recall Curve
The model maintains a high precision plateau, allowing the business to catch up to 70% of defaults with almost zero false alarms.

### Tech Stack 
* Language: Python
* Core Libraries: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE/Pipeline)
* Preprocessing: ColumnTransformer, StandardScaler, OneHotEncoder
