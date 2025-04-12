**Project Overview**

This project is created to compare 5 different classification models. The dataset which is considered for this task includes data from a bank transaction history. 
The transactions are categorised as Fraud or not in 'isFraud' column as 0 and 1. 0 means not fraud and 1 means fraudulent transaction.
The isFlaggedFraud column can be ignored as it is flagged fraud from a rule engine for given set of rules.

**Key Features & Technologies**

In this project we have considered below classsification models to compare.

1. Logistic Regression
2. DecisionTreeClassifier
3. RandomForestClassifier
4. GaussianNB
5. XGBClassifier

For measure the performance of each model, we have considered few metrics like Accuracy, Recall, Precision, F1 Score and AUC.
   
**Setup Instructions**

For the project to run we need below libraries to be installed.

1. imblearn - Used for balacing the imbalaced data which is highly skewed.
2. Pandas - For reading data from input file
3. Numpy - For numerical analysis
4. Seaborn - For Grphical representation of data
5. Plotly - For Grphical representation of data
6. Scikit-learn - For ML models and Preprocessing

Additionally we also need the input data which can be found here.

https://drive.google.com/file/d/10BRa-fCyVd4K3L9iLWQC8vp2DWJAiKSV/view?usp=drive_link
