# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
### Importing dataset

# %%
file_path = 'C:/Users/sidde/anaconda3/envs/FraudDetection/source/Insurance data.csv'
df = pd.read_csv(file_path)
df.head()

# %%
df.columns

# %% [markdown]
# ### EDA
# 

# %% [markdown]
# #### 1. Handling missing values
# 
# 

# %%
df.isna().sum()

# %%
# Replace NaN values with 0's for marital_status,witness_present_ind as it has binary values
df['marital_status'].fillna(0, inplace=True)
df['witness_present_ind'].fillna(0, inplace=True)

# Replace NaN values with mean values for claim_est_payout,age_of_vehicle as it has continuous values
df['claim_est_payout'].fillna(df['claim_est_payout'].median(), inplace=True)
df['age_of_vehicle'].fillna(df['age_of_vehicle'].median(), inplace=True)



# %% [markdown]
# #### Cleaning independent variables

# %%
median_age = df['age_of_driver'].median()
# Replace ages greater than 100 with the median age
df['age_of_driver'] = np.where(df['age_of_driver'] > 100, median_age, df['age_of_driver'])


median_income = df['annual_income'].median()
# Replace ages greater than 100 with the median age
df['annual_income'] = np.where(df['annual_income'] < 0, median_income, df['annual_income'])

# %% [markdown]
# #### 2. Cleaning target variable

# %%
# Checking the target variable
df["fraud"].value_counts()


# %%
# Target variable fraud has -1,0,1 values where it can have only 0's and 1's. So dropping the outliers
df = df[df["fraud"] != -1]

# %% [markdown]
# #### 3. Handling temporal values
# 

# %%
# Convert the 'Date' column to a datetime object
df['claim_date'] = pd.to_datetime(df['claim_date'])
# Extract the year and create a new 'Year' column
df['Claim_Year'] = df['claim_date'].dt.year



# %%


# %% [markdown]
# ### Feature Engineering

# %%
df = df.drop('claim_date', axis=1)

df = df.drop('zip_code', axis=1)
#Zip code & claim_date is dropped as its not useful for classification.Instead of claim_date, Claim_Year will be suitable.

df = df.drop('claim_number', axis=1)
#Claim Number is a unique column, hence removed.

# %% [markdown]
# #### Numerical Variables

# %%
numerical_features=[feature for feature in df.columns if df[feature].dtype!='O']

# %%
numerical_features

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Extract numerical columns (you might need to adapt this based on your DataFrame structure)
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

# Apply Min-Max scaling to the numerical columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# %% [markdown]
# #### Categorical Variables

# %%
categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']

for feature in categorical_features[:]:
    print(feature,":",len(df[feature].unique()),'labels')

# %% [markdown]
# The Vehicle_color has more than 3 labels. Hence,
# Frequency encoding : vehicle_color , 
# One hot encoding : gender,living_status,accident_site,channel,vehicle_category,
# Ordinal Number Encoding : claim_day_of_week

# %% [markdown]
# ###### Ordinal Number Encoding

# %%
dictionary={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

#Encodes all the day names to assigned numbers
df['claim_day_of_week']=df['claim_day_of_week'].map(dictionary)


# %% [markdown]
# ######  Frequency Encoding

# %%
Map = df['vehicle_color'].value_counts().to_dict()

#Mapping the respective value counts to the colors
df['vehicle_color']=df['vehicle_color'].map(Map)


# %% [markdown]
# ######  One Hot Encoding

# %%
# Create dummy columns
accident_site_dummies = pd.get_dummies(df['accident_site'], drop_first=True)
channel_dummies = pd.get_dummies(df['channel'], drop_first=True)
vehicle_category_dummies = pd.get_dummies(df['vehicle_category'], drop_first=True)

# Convert boolean values to integers (0 or 1)
accident_site_dummies = accident_site_dummies.astype(int)
channel_dummies = channel_dummies.astype(int)
vehicle_category_dummies = vehicle_category_dummies.astype(int)

# Concatenate the dummy columns with the original DataFrame
df = pd.concat([df, accident_site_dummies], axis=1)
df = pd.concat([df, channel_dummies], axis=1)
df = pd.concat([df, vehicle_category_dummies], axis=1)

# Drop the original columns
df = df.drop(['accident_site', 'channel', 'vehicle_category'], axis=1)

# %% [markdown]
# ######  Binary Encoding

# %%
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['living_status'] = df['living_status'].map({'Rent': 1, 'Own': 0})


# %%


# %% [markdown]
# #### Train Test Split

# %%
from sklearn.model_selection import train_test_split, GridSearchCV

# Assume 'df' is your DataFrame with features and target variable
# X should contain the features (independent variables), and y should contain the target variable (dependent variable)

# Assuming 'fraud' is the target variable, and the rest are features
X = df.drop('fraud', axis=1)
y = df['fraud']

# Split the data into training and testing sets
# Adjust the 'test_size' parameter as needed (e.g., test_size=0.3 for an 70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shapes of the resulting sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# %% [markdown]
# ### Data Modelling

# %%
#!pip install xgboost
# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import sys

# %%


from imblearn.combine import SMOTETomek
from collections import Counter

# Assuming X_train and y_train are your training data
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

print("The number of classes before fit: {}".format(Counter(y_train)))
print("The number of classes after fit: {}".format(Counter(y_train_resampled)))


# %% [markdown]
# ## MLFlow

# %%
# Create a pipeline with imputation and XGBClassifier
xgb_pipeline = Pipeline([

    ('classifier', XGBClassifier(random_state=42))
])



# Get parameters from command line or use default values
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
min_child_weight = int(sys.argv[4]) if len(sys.argv) > 4 else 1
subsample = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
colsample_bytree = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0


# Create a pipeline with preprocessor and XGBClassifier with dynamic parameters
xgb_pipeline = Pipeline([

    ('classifier', XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    ))
])

# Fit the pipeline
with mlflow.start_run():
    xgb_pipeline.fit(X_train_resampled, y_train_resampled)

    # Predictions
    y_pred_xgb = xgb_pipeline.predict(X_test)


    # Compute precision, recall, and F1 score
    precision_xgb = precision_score(y_test, y_pred_xgb)
    recall_xgb = recall_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)

    # Log precision, recall, and F1 score
    mlflow.log_metric("precision", precision_xgb)
    mlflow.log_metric("recall", recall_xgb)
    mlflow.log_metric("f1_score", f1_xgb)

    # Log confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
    mlflow.log_metric("true_negatives", tn)
    mlflow.log_metric("false_positives", fp)
    mlflow.log_metric("false_negatives", fn)
    mlflow.log_metric("true_positives", tp)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_child_weight", min_child_weight)
    mlflow.log_param("subsample", subsample)
    mlflow.log_param("colsample_bytree", colsample_bytree)

    # Log the model
    mlflow.sklearn.log_model(xgb_pipeline, "xgboost_model")

    # Print classification report and confusion matrix
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb))

    print("\nXGBoost Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))