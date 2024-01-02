import tempfile
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow
import os


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

    median_income = df['annual_income'].median()
    # Replace ages greater than 100 with the median age
    df['annual_income'] = np.where(df['annual_income'] < 0, median_income, df['annual_income'])

    # Convert the 'Date' column to a datetime object
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    # Extract the year and create a new 'Year' column
    df['Claim_Year'] = df['claim_date'].dt.year

    df = df.drop('claim_date', axis=1)

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


def process_excel(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        preprocessed_df = preprocess(df)

        # Load the model from the pickle file
        logged_model = 'runs:/2297385644c347eaad5053a7eb4edbcf/xgboost_model'
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on the preprocessed DataFrame
        df['Fraud'] = loaded_model.predict(preprocessed_df)

        # Display the DataFrame
        st.write("Processed Data:")
        st.write(df.head())

        # Save the result to an Excel file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            processed_file_path = temp_file.name
            df.to_excel(processed_file_path, index=False)

        return processed_file_path

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.set_page_config(page_title="DISSERTATION")

    st.header("FRAUDULENT CLAIMS DETECTION MODEL")

    # File upload and processing logic
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        st.write("File successfully uploaded!")

        # Display the uploaded file content
        st.write("Uploaded File Content:")
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write(df)

        # Button to trigger the processing
        if st.button("Predict Fraudulent"):
            try:
                # Save the uploaded file temporarily
                temp_file_path = process_excel(uploaded_file)

                # Display the processed file link
                if temp_file_path:
                    st.markdown(f"### [Download Processed File]({temp_file_path})")

            finally:
                # Cleanup: Remove the temporary file
                os.remove(temp_file_path)

    st.text("Note: The processed Excel file will be saved with a temporary name.")

if __name__ == "__main__":
    main()


##New