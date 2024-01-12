from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow

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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        # Extracting values from the form
        data = {
            'claim_number': int(request.form['claim_number']),
            'age_of_driver': int(request.form['age_of_driver']),
            'gender': request.form['gender'],
            'marital_status': int(request.form['marital_status']),
            'safty_rating': int(request.form['safty_rating']),
            'annual_income': float(request.form['annual_income']),
            'high_education_ind': int(request.form['high_education_ind']),
            'address_change_ind': int(request.form['address_change_ind']),
            'living_status': request.form['living_status'],
            'zip_code': int(request.form['zip_code']),
            'claim_date': request.form['claim_date'],
            'claim_day_of_week': request.form['claim_day_of_week'],
            'accident_site': request.form['accident_site'],
            'past_num_of_claims': int(request.form['past_num_of_claims']),
            'witness_present_ind': int(request.form['witness_present_ind']),
            'liab_prct': int(request.form['liab_prct']),
            'channel': request.form['channel'],
            'policy_report_filed_ind': int(request.form['policy_report_filed_ind']),
            'claim_est_payout': float(request.form['claim_est_payout']),
            'age_of_vehicle': int(request.form['age_of_vehicle']),
            'vehicle_category': request.form['vehicle_category'],
            'vehicle_price': float(request.form['vehicle_price']),
            'vehicle_color': request.form['vehicle_color'],
            'vehicle_weight': float(request.form['vehicle_weight']),
        }

        # Create a DataFrame
        df = pd.DataFrame(data, index=[0])

        # Preprocess the data
        preprocessed_df = preprocess(df)

        # Load the model from the pickle file
        logged_model = 'runs:/fd60b73ca0174482bd377ba4317a2d1e/adaboost_model'
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Ensure dummy columns match between training and test datasets
        dummy_columns = set(preprocessed_df.columns) - {'Fraud'}  # Exclude the target column
        missing_columns = set(loaded_model.input_names) - dummy_columns

        for col in missing_columns:
            preprocessed_df[col] = 0  # Add missing dummy columns with default value

        # Predict on the preprocessed DataFrame
        predictions = loaded_model.predict(preprocessed_df)

        # Add the predicted Fraud column to the DataFrame
        preprocessed_df['Fraud'] = predictions

        # Display the prediction
        fraud_prediction = "Fraud" if predictions[0] == 1 else "No Fraud"
        
        return render_template('fraud_output.html', prediction=fraud_prediction)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)