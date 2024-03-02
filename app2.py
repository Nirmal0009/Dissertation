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
        logged_model = 'runs:/3a1b33a59ea14124ae6827c354e3c213/adaboost_model'
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on the preprocessed DataFrame
        prediction = loaded_model.predict(df)

        return prediction

    except Exception as e:
        return {"error": str(e)}

st.title("Fraud Detection App")

st.subheader("Enter column values:")

age_of_driver_input = st.text_input('Enter value for age_of_driver', key='age_of_driver')

# Gender Dropdown
gender_input = st.selectbox('Select gender', ['M', 'F'], key='gender')

# Marital Status Dropdown
marital_status_input = st.selectbox('Select marital status', ['0', '1'], key='marital_status')

safty_rating_input = st.text_input('Enter value for safty_rating', key='safty_rating')
annual_income_input = st.text_input('Enter value for annual_income', key='annual_income')
high_education_ind_input = st.selectbox('Select high_education_ind', ['0', '1'], key='high_education_ind')
address_change_ind_input = st.selectbox('Select address_change_ind', ['0', '1'], key='address_change_ind')

# Living Status Dropdown
living_status_input = st.selectbox('Select living status', ['Rent', 'Own'], key='living_status')

# Accident Site Dropdown
accident_site_input = st.selectbox('Select accident site', ['Local', 'Highway', 'Parking lot'], key='accident_site')

past_num_of_claims_input = st.text_input('Enter value for past_num_of_claims', key='past_num_of_claims')
witness_present_ind_input = st.selectbox('Select witness present', ['0', '1'], key='witness_present_ind')
liab_prct_input = st.text_input('Enter value for liab_prct', key='liab_prct')
channel_input = st.selectbox('Select channel', ['Online', 'Broker', 'Phone'], key='channel')
policy_report_filed_ind_input = st.selectbox('Select policy_report_filed_ind', ['0', '1'], key='policy_report_filed_ind')
claim_est_payout_input = st.text_input('Enter value for claim_est_payout', key='claim_est_payout')
age_of_vehicle_input = st.text_input('Enter value for age_of_vehicle', key='age_of_vehicle')

# Vehicle Category Dropdown
vehicle_category_input = st.selectbox('Select vehicle category', ['Large', 'Medium', 'Compact'], key='vehicle_category')
vehicle_price_input = st.text_input('Enter value for vehicle_price', key='vehicle_price')

# Vehicle Color Dropdown
vehicle_color_input = st.selectbox('Select vehicle color', ['silver', 'other', 'black', 'white', 'red', 'blue', 'gray'], key='vehicle_color')
vehicle_weight_input = st.text_input('Enter value for vehicle_weight', key='vehicle_weight')

claim_date_input = st.text_input('Enter claim date (YYYY-MM-DD)', key='claim_date')

# Claim Day of Week Dropdown
claim_day_of_week_input = st.selectbox('Select claim day of week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key='claim_day_of_week')

if st.button("Predict"):
    input_data = {
        'age_of_driver': [float(age_of_driver_input) if age_of_driver_input else np.nan],
        'gender': [gender_input],
        'marital_status': [int(marital_status_input)],
        'safty_rating': [float(safty_rating_input) if safty_rating_input else np.nan],
        'annual_income': [float(annual_income_input) if annual_income_input else np.nan],
        'high_education_ind': [int(high_education_ind_input)],
        'address_change_ind': [int(address_change_ind_input)],
        'living_status': [living_status_input],
        'claim_date': [claim_date_input],
        'claim_day_of_week': [claim_day_of_week_input],
        'accident_site': [accident_site_input],
        'past_num_of_claims': [float(past_num_of_claims_input) if past_num_of_claims_input else np.nan],
        'witness_present_ind': [int(witness_present_ind_input)],
        'liab_prct': [float(liab_prct_input) if liab_prct_input else np.nan],
        'channel': [channel_input],
        'policy_report_filed_ind': [int(policy_report_filed_ind_input)],
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
        st.write("Prediction: Not a fraud")

    elif prediction_result == 1:
        st.write("Prediction: Fraud")

        
