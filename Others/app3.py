from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

app = Flask(__name__)

def preprocess(df):
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

    
    # Handling temporal values
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    df['Claim_Year'] = df['claim_date'].dt.year

    # Feature Engineering
    df = df.drop(['claim_date', 'zip_code', 'claim_number'], axis=1)

    # Numerical Variables scaling
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Categorical Variables encoding
    dictionary = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df['claim_day_of_week'] = df['claim_day_of_week'].map(dictionary)

    map_vehicle_color = df['vehicle_color'].value_counts().to_dict()
    df['vehicle_color'] = df['vehicle_color'].map(map_vehicle_color)

    df = pd.get_dummies(df, columns=['accident_site', 'channel', 'vehicle_category'], drop_first=True)

    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    df['living_status'] = df['living_status'].map({'Rent': 1, 'Own': 0})

    return df


def align_columns(df, model_columns):
    # Ensure dummy columns match between training and test datasets
    dummy_columns = set(df.columns)  # Include all columns
    missing_columns = dummy_columns - set(model_columns)

    for col in missing_columns:
        df[col] = 0  # Add missing dummy columns with default value

    return df


@app.route('/')
def index():
    # Default values for the form fields
    default_values = {
        'claim_number': 0,
        'age_of_driver': 39,
        'gender': 'M',
        'marital_status': 1,
        'safty_rating': 0,
        'annual_income': 36633,
        'high_education_ind': 0,
        'address_change_ind': 0,
        'living_status': 'Own',
        'zip_code': 50048,
        'claim_date': '08-12-2016',
        'claim_day_of_week': 'Friday',
        'accident_site': 'Highway',
        'past_num_of_claims': 0,
        'witness_present_ind': 0,
        'liab_prct': 25,
        'channel': 'Phone',
        'policy_report_filed_ind': 0,
        'claim_est_payout': 5196.552552,
        'age_of_vehicle': 25,
        'vehicle_category': 'Large',
        'vehicle_price': 24360.59273,
        'vehicle_color': 'silver',
        'vehicle_weight': 26633.27819,
    }

    return render_template('index.html', default_values=default_values)


@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        # Extracting values from the form
        data = {
            'claim_number': int(request.form.get('claim_number', 0)),
            'age_of_driver': int(request.form.get('age_of_driver', 39)),
            'gender': request.form.get('gender', 'M'),
            'marital_status': int(request.form.get('marital_status', 1)),
            'safty_rating': int(request.form.get('safty_rating', 0)),
            'annual_income': float(request.form.get('annual_income', 36633)),
            'high_education_ind': int(request.form.get('high_education_ind', 0)),
            'address_change_ind': int(request.form.get('address_change_ind', 0)),
            'living_status': request.form.get('living_status', 'Own'),
            'zip_code': int(request.form.get('zip_code', 50048)),
            'claim_date': request.form.get('claim_date', '08-12-2016'),
            'claim_day_of_week': request.form.get('claim_day_of_week', 'Friday'),
            'accident_site': request.form.get('accident_site', 'Highway'),
            'past_num_of_claims': int(request.form.get('past_num_of_claims', 0)),
            'witness_present_ind': int(request.form.get('witness_present_ind', 0)),
            'liab_prct': int(request.form.get('liab_prct', 25)),
            'channel': request.form.get('channel', 'Phone'),
            'policy_report_filed_ind': int(request.form.get('policy_report_filed_ind', 0)),
            'claim_est_payout': float(request.form.get('claim_est_payout', 5196.552552)),
            'age_of_vehicle': int(request.form.get('age_of_vehicle', 25)),
            'vehicle_category': request.form.get('vehicle_category', 'Large'),
            'vehicle_price': float(request.form.get('vehicle_price', 24360.59273)),
            'vehicle_color': request.form.get('vehicle_color', 'silver'),
            'vehicle_weight': float(request.form.get('vehicle_weight', 26633.27819)),
        }

  # Create a DataFrame
 # Create a DataFrame
        df = pd.DataFrame(data, index=[0])

        # Preprocess the data
        preprocessed_df = preprocess(df)

        # Load the model from the pickle file
        logged_model_path = 'runs:/fd60b73ca0174482bd377ba4317a2d1e/adaboost_model'
        loaded_model = mlflow.sklearn.load_model(logged_model_path)

        # Use the feature names directly from the columns
        model_input_columns = preprocessed_df.columns

        # Ensure dummy columns match between training and test datasets
        preprocessed_df = align_columns(preprocessed_df, model_input_columns)

        # Predict on the preprocessed DataFrame
        predictions = loaded_model.predict(preprocessed_df)

        # Display the prediction
        fraud_prediction = "Fraud" if predictions[0] == 1 else "No Fraud"
        
        return render_template('fraud_output.html', prediction=fraud_prediction)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)