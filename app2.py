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
    # Replace NaN values with 0's for marital_status, witness_present_ind as it has binary values
    df['marital_status'].fillna(0, inplace=True)
    df['witness_present_ind'].fillna(0, inplace=True)

    # Replace NaN values with mean values for claim_est_payout, age_of_vehicle as it has continuous values
    df['claim_est_payout'].fillna(df['claim_est_payout'].median(), inplace=True)
    df['age_of_vehicle'].fillna(df['age_of_vehicle'].median(), inplace=True)
    
    median_age = df['age_of_driver'].median()
    # Replace ages greater than 100 with the median age
    df['age_of_driver'] = np.where(df['age_of_driver'] > 100, median_age, df['age_of_driver'])
    
    # Replace ages greater than 100 with the median age
    df['age_of_driver'] = np.where(df['age_of_driver'] > 100, median_age, df['age_of_driver'])
    median_income = df['annual_income'].median()

    # Replace ages greater than 100 with the median age
    df['annual_income'] = np.where(df['annual_income'] < 0, median_income, df['annual_income'])

    # Convert the 'Date' column to a datetime object
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    # Extract the year and create a new 'Year' column
    df['Claim_Year'] = df['claim_date'].dt.year

    df = df.drop('claim_date', axis=1)#
    df = df.drop('zip_code', axis=1)
    # Zip code & claim_date is dropped as its not useful for classification. Instead of claim_date, Claim_Year will be suitable.
    df = df.drop('claim_number', axis=1)
    # Claim Number is a unique column, hence removed.

    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    dictionary = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df['claim_day_of_week'] = df['claim_day_of_week'].map(dictionary)

    Map = df['vehicle_color'].value_counts().to_dict()
    df['vehicle_color'] = df['vehicle_color'].map(Map)
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
            logged_model = 'runs:/8aa697a99f38480292e6106b03d63334/adaboost_model'
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
