from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow
import os

app = Flask(__name__, static_folder='static')

# Ensure 'uploads' folder exists
uploads_folder = os.path.join(os.getcwd(), 'uploads')
os.makedirs(uploads_folder, exist_ok=True)

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

@app.route('/process_excel', methods=['POST'])
def process_excel():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    # If the user does not select a file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file'

    if file and file.filename.endswith(('.xlsx', '.xls')):
        try:
            # Save the file to a temporary location
            temp_path = os.path.join(uploads_folder, file.filename)
            file.save(temp_path)

            # Read the Excel file
            df = pd.read_excel(temp_path, engine='openpyxl')
            preprocessed_df = preprocess(df)

            # Load the model from the pickle file
            logged_model = 'runs:/09432b6ca4dd47658b6358a8e1b3df76/adaboost_model'
            loaded_model = mlflow.pyfunc.load_model(logged_model)

            # Predict on the preprocessed DataFrame
            preprocessed_df['Fraud'] = loaded_model.predict(preprocessed_df)

            # Add the predicted 'Fraud' column to the original DataFrame
            df['Predicted_Fraud'] = preprocessed_df['Fraud']

            # Save the new DataFrame to a new Excel file
            output_file_path = os.path.join(uploads_folder, f"output_{file.filename}")
            df.to_excel(output_file_path, index=False)

            # Remove the temporary file
            os.remove(temp_path)

            return f"Predictions added to the file: {output_file_path}"

        except Exception as e:
            return f"Error: {e}"

    else:
        return "Please select a valid Excel file."


if __name__ == '__main__':
    app.run(debug=True)