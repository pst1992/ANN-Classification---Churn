import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

## Streamlit app
st.title("Customer Churn Prediction")

# user input
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.number_input("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale the input data
input_data_scaled = scaler.transform(input_data)

## Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write(f"Customer is likely to churn with a probability of {prediction_proba:.2f}")
else:
    st.write(f"Customer is unlikely to churn with a probability of {1 - prediction_proba:.2f}")