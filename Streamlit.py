import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow
import streamlit as st
import time

def run_prediction(df):
    try:
        # Convert numeric columns to appropriate data types
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
        df = df.drop(['claim_date'], axis=1)

        # Numerical Variables scaling
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = MinMaxScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

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

        # Load the model from the pickle file
        logged_model = 'runs:/c5f3b97a18cb48ba9bdfebe28c331d52/adaboost_model'
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on the preprocessed DataFrame
        prediction = loaded_model.predict(df)

        return prediction

    except Exception as e:
        return {"error": str(e)}



st.title("FRAUDULENT CLAIMS DETECTION MODEL")

st.subheader("Enter column values:")

# Age of Driver Text Input
age_of_driver_input, adjacent_col = st.columns(2)
with age_of_driver_input:
    age_of_driver_input = st.text_input('Age of Driver', key='age_of_driver')
with adjacent_col:
    gender_input = st.selectbox('Gender', ['M', 'F'], key='gender')

# Marital Status Dropdown
marital_status_input, adjacent_col = st.columns(2)
with marital_status_input:
    marital_status_input = st.selectbox('Marital Status', ['No', 'Yes'], key='marital_status')
with adjacent_col:
    safty_rating_input = st.text_input('Safety Rating', key='safty_rating')

# Annual Income Text Input
annual_income_input, adjacent_col = st.columns(2)
with annual_income_input:
    annual_income_input = st.text_input('Annual Income', key='annual_income')
with adjacent_col:
    high_education_ind_input = st.selectbox('High Education', ['No', 'Yes'], key='high_education_ind')

# Address Change Ind Dropdown
address_change_ind_input, adjacent_col = st.columns(2)
with address_change_ind_input:
    address_change_ind_input = st.selectbox('Address Change', ['No', 'Yes'], key='address_change_ind')
with adjacent_col:
    living_status_input = st.selectbox('Living Status', ['Rent', 'Own'], key='living_status')

# Accident Site Dropdown
accident_site_input, adjacent_col = st.columns(2)
with accident_site_input:
    accident_site_input = st.selectbox('Accident Site', ['Local', 'Highway', 'Parking lot'], key='accident_site')
with adjacent_col:
    past_num_of_claims_input = st.text_input('Past Number of Claims', key='past_num_of_claims')

# Witness Present Ind Dropdown
witness_present_ind_input, adjacent_col = st.columns(2)
with witness_present_ind_input:
    witness_present_ind_input = st.selectbox('Witness Present', ['No', 'Yes'], key='witness_present_ind')
with adjacent_col:
    liab_prct_input = st.text_input('Liability Percentage', key='liab_prct')

# Channel Dropdown
channel_input, adjacent_col = st.columns(2)
with channel_input:
    channel_input = st.selectbox('Channel', ['Online', 'Broker', 'Phone'], key='channel')
with adjacent_col:
    policy_report_filed_ind_input = st.selectbox('Policy Report Filed', ['No', 'Yes'], key='policy_report_filed_ind')

# Claim Est Payout Text Input
claim_est_payout_input, adjacent_col = st.columns(2)
with claim_est_payout_input:
    claim_est_payout_input = st.text_input('Claim Est Payout', key='claim_est_payout')
with adjacent_col:
    age_of_vehicle_input = st.text_input('Age of Vehicle', key='age_of_vehicle')

# Vehicle Category Dropdown
vehicle_category_input, adjacent_col = st.columns(2)
with vehicle_category_input:
    vehicle_category_input = st.selectbox('Vehicle Category', ['Large', 'Medium', 'Compact'], key='vehicle_category')
with adjacent_col:
    vehicle_price_input = st.text_input('Vehicle Price', key='vehicle_price')

# Vehicle Color Dropdown
vehicle_color_input, adjacent_col = st.columns(2)
with vehicle_color_input:
    vehicle_color_input = st.selectbox('Vehicle Color', ['silver', 'other', 'black', 'white', 'red', 'blue', 'gray'], key='vehicle_color')
with adjacent_col:
    vehicle_weight_input = st.text_input('Vehicle Weight', key='vehicle_weight')

# Claim Date Text Input
claim_date_input, adjacent_col = st.columns(2)
with claim_date_input:
    claim_date_input = st.text_input('Claim Date (YYYY-MM-DD)', key='claim_date')
with adjacent_col:
    claim_day_of_week_input = st.selectbox('Claim Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key='claim_day_of_week')

if st.button("Predict"):
    input_data = {
        'age_of_driver': [float(age_of_driver_input) if age_of_driver_input else np.nan],
        'gender': [gender_input],
        'marital_status': [1 if marital_status_input == 'Yes' else 0],
        'safty_rating': [float(safty_rating_input) if safty_rating_input else np.nan],
        'annual_income': [float(annual_income_input) if annual_income_input else np.nan],
        'high_education_ind': [1 if high_education_ind_input == 'Yes' else 0],
        'address_change_ind': [1 if address_change_ind_input == 'Yes' else 0],
        'living_status': [living_status_input],
        'claim_date': [claim_date_input],
        'claim_day_of_week': [claim_day_of_week_input],
        'accident_site': [accident_site_input],
        'past_num_of_claims': [float(past_num_of_claims_input) if past_num_of_claims_input else np.nan],
        'witness_present_ind': [1 if witness_present_ind_input == 'Yes' else 0],
        'liab_prct': [float(liab_prct_input) if liab_prct_input else np.nan],
        'channel': [channel_input],
        'policy_report_filed_ind': [1 if policy_report_filed_ind_input == 'Yes' else 0],
        'claim_est_payout': [float(claim_est_payout_input) if claim_est_payout_input else np.nan],
        'age_of_vehicle': [float(age_of_vehicle_input) if age_of_vehicle_input else np.nan],
        'vehicle_category': [vehicle_category_input],
        'vehicle_price': [float(vehicle_price_input) if vehicle_price_input else np.nan],
        'vehicle_color': [vehicle_color_input],
        'vehicle_weight': [float(vehicle_weight_input) if vehicle_weight_input else np.nan],
    }

    input_df = pd.DataFrame(input_data)

    prediction_result = run_prediction(input_df)
    if prediction_result == 0:
        st.markdown("<p style='font-size:24px; color:green'>Prediction indicates no fraudulent activity detected </p>", unsafe_allow_html=True)
    elif prediction_result == 1:
        st.markdown("<p style='font-size:24px; color:red'>Prediction indicates fraudulent activity </p>", unsafe_allow_html=True)
#new
