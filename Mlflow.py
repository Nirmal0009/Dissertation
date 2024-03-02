# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import mlflow
import sys
import mlflow.sklearn
from imblearn.combine import SMOTETomek


### Importing dataset
file_path = r'C:\Users\sidde\OneDrive\Documents\Dissertation\Final\Data_source\Insurance data.csv'
df = pd.read_csv(file_path)




def preprocess_data(df):
    # Handling missing values
    df['marital_status'].fillna(0, inplace=True)
    df['witness_present_ind'].fillna(0, inplace=True)

    df['claim_est_payout'].fillna(df['claim_est_payout'].median(), inplace=True)
    df['age_of_vehicle'].fillna(df['age_of_vehicle'].median(), inplace=True)

    median_age = df['age_of_driver'].median()
    Q1 = df['age_of_driver'].quantile(0.25)
    Q3 = df['age_of_driver'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['age_of_driver'] = np.where((df['age_of_driver'] < lower_bound) | (df['age_of_driver'] > upper_bound),
                                    median_age, df['age_of_driver'])

    median_income = df['annual_income'].median()
    df['annual_income'] = np.where(df['annual_income'] < 0, median_income, df['annual_income'])

    # Cleaning target variable
    df = df[df["fraud"] != -1]

    # Handling temporal values
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    df['Claim_Year'] = df['claim_date'].dt.year

    # Feature Engineering
    df = df.drop(['claim_date', 'zip_code', 'claim_number'], axis=1)

# Numerical Variables scaling
    # numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    # scaler = MinMaxScaler()
    # df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Categorical Variables encoding
    dictionary = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df['claim_day_of_week'] = df['claim_day_of_week'].map(dictionary)


    dictionary1 = {'black': 1, 'silver': 2, 'white': 3, 'red': 4, 'blue': 5, 'gray': 6, 'other': 7}
    df['vehicle_color'] = df['vehicle_color'].map(dictionary1)

    dictionary2 = {'Compact': 1, 'Large': 2, 'Medium': 3}
    df['vehicle_category'] = df['vehicle_category'].map(dictionary2)

    dictionary3 = {'Broker': 1, 'Phone': 2, 'Online': 3}
    df['channel'] = df['channel'].map(dictionary3)

    dictionary4 = {'Local': 1, 'Parking Lot': 2, 'Highway': 3}
    df['accident_site'] = df['accident_site'].map(dictionary4)

    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    df['living_status'] = df['living_status'].map({'Rent': 1, 'Own': 0})


    return df

df = preprocess_data(df)




#### Train Test Split
X = df.drop('fraud', axis=1)
y = df['fraud']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shapes of the resulting sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)






#Handling Imbalance Data using SMote tomek
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

print("The number of classes before fit: {}".format(Counter(y_train)))
print("The number of classes after fit: {}".format(Counter(y_train_resampled)))





## MLFlow
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50  
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  

ada_pipeline = Pipeline([
    ('classifier', AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),  # DecisionTreeClassifier is the default base estimator for AdaBoost
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    ))
])

# Fit the pipeline
with mlflow.start_run():
    ada_pipeline.fit(X_train_resampled, y_train_resampled)

    # Predictions
    y_pred_ada = ada_pipeline.predict(X_test)

    # Compute precision, recall, and F1 score
    precision_ada = precision_score(y_test, y_pred_ada)
    recall_ada = recall_score(y_test, y_pred_ada)
    f1_ada = f1_score(y_test, y_pred_ada)

    # Log precision, recall, and F1 score
    mlflow.log_metric("precision", precision_ada)
    mlflow.log_metric("recall", recall_ada)
    mlflow.log_metric("f1_score", f1_ada)

    # Log confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ada).ravel()
    mlflow.log_metric("true_negatives", tn)
    mlflow.log_metric("false_positives", fp)
    mlflow.log_metric("false_negatives", fn)
    mlflow.log_metric("true_positives", tp)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)

    # Log the model
    mlflow.sklearn.log_model(ada_pipeline, "adaboost_model")

    # Print classification report and confusion matrix
    print("\nAdaBoost Classification Report:")
    print(classification_report(y_test, y_pred_ada))

    print("\nAdaBoost Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_ada))
# %%
