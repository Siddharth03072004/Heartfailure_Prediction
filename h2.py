#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model
with open('rf_model_heart.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🫀 Heart Failure Prediction System")
st.markdown("Predict the risk of heart failure using clinical parameters.")

st.sidebar.header("🧾 Enter Patient Data")

# Collect user input
def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 50)
    anaemia = st.sidebar.radio('Anaemia', [0, 1])
    creatinine_phosphokinase = st.sidebar.slider('Creatinine Phosphokinase', 20, 8000, 250)
    diabetes = st.sidebar.radio('Diabetes', [0, 1])
    ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', 10, 70, 38)
    high_blood_pressure = st.sidebar.radio('High Blood Pressure', [0, 1])
    platelets = st.sidebar.slider('Platelets (k/mL)', 100000, 600000, 250000)
    serum_creatinine = st.sidebar.slider('Serum Creatinine', 0.5, 10.0, 1.5)
    serum_sodium = st.sidebar.slider('Serum Sodium', 110, 150, 138)
    sex = st.sidebar.radio('Sex', [0, 1])
    smoking = st.sidebar.radio('Smoking', [0, 1])
    time = st.sidebar.slider('Follow-up Time (days)', 0, 300, 120)

    data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()
st.subheader("📊 Patient Data Preview")
st.write(df)

# Prediction
if st.button('🔍 Predict'):
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]

    st.subheader("🧠 Prediction Result")
    st.metric("Heart Failure Risk", "Yes" if prediction == 1 else "No")

    # Probability Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prediction_proba[1] * 100, 2),
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if prediction == 1 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ]
        },
        title={'text': "Risk Probability (%)"},
    ))
    st.plotly_chart(fig_gauge)

    # Bar Chart for probabilities
    fig_bar = px.bar(
        x=["No Heart Failure", "Heart Failure"],
        y=prediction_proba,
        labels={"x": "Outcome", "y": "Probability"},
        text=np.round(prediction_proba, 2),
        color=["No Heart Failure", "Heart Failure"],
        color_discrete_sequence=["green", "red"]
    )
    st.subheader("📈 Prediction Probability Chart")
    st.plotly_chart(fig_bar, use_container_width=True)


# In[ ]:




