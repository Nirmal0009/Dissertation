import tempfile
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow
import os


def preprocess_data(df):
 
    # Handling temporal values
    df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce', format='%m/%d/%Y')
    df['Claim_Year'] = df['claim_date'].dt.year

    # Handling missing values
    df['marital_status'].fillna(0, inplace=True)
    df['witness_present_ind'].fillna(0, inplace=True)

    df['claim_est_payout'].fillna(df['claim_est_payout'].median(), inplace=True)
    df['age_of_vehicle'].fillna(df['age_of_vehicle'].median(), inplace=True)

    median_age = df['age_of_driver'].median()
    # Replace ages greater than 100 with the median age
    df['age_of_driver'] = np.where(df['age_of_driver'] > 100, median_age, df['age_of_driver'])

    median_income = df['annual_income'].median()
    df['annual_income'] = np.where(df['annual_income'] < 0, median_income, df['annual_income'])

    # Cleaning target variable
    df = df[df["fraud"] != -1]

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


def process_excel(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        preprocessed_df = preprocess_data(df)

        # Load the model from the pickle file
        logged_model = 'runs:/8aa697a99f38480292e6106b03d63334/adaboost_model'
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on the preprocessed DataFrame
        df['fraud'] = loaded_model.predict(preprocessed_df).astype(int)
     
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
            temp_file_path = None
            try:
                # Save the uploaded file temporarily
                temp_file_path = process_excel(uploaded_file)

                # Display the processed file link
                if temp_file_path:
                    st.markdown(f"### [Download Processed File]({temp_file_path})")

            except Exception as e:
                st.error(f"Error: {e}")

            finally:
                # Cleanup: Remove the temporary file if it exists
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    st.text("Note: The processed Excel file will be saved with a temporary name.")

if __name__ == "__main__":
    main()
