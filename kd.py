import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

model = pickle.load(open('kd.pkl', 'rb'))

with st.sidebar:
    selected = option_menu("Choose Prediction System", ['KidneyPrediction', 'HeartPrediction'])

if selected == 'KidneyPrediction':
    st.title('Kidney Prediction using ML')
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input('Age', min_value=0)
        blood_pressure = st.number_input('Blood Pressure', min_value=0)
        specific_gravity = st.number_input('Specific Gravity', min_value=0.0, max_value=2.0, step=0.1)
        albumin = st.number_input('Albumin', min_value=0)
        sugar = st.number_input('Sugar', min_value=0)
        red_blood_cells = st.number_input('Red Blood Cells', min_value=0)

    with col2:
        pus_cell = st.number_input('Pus Cell', min_value=0)
        pus_cell_clumps = st.number_input('Pus Cell Clumps', min_value=0)
        bacteria = st.number_input('Bacteria', min_value=0)
        blood_glucose_random = st.number_input('Blood Glucose Random', min_value=0)
        blood_urea = st.number_input('Blood Urea', min_value=0)
        serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0, step=0.1)

    with col3:
        sodium = st.number_input('Sodium', min_value=0)
        potassium = st.number_input('Potassium', min_value=0.0, step=0.1)
        hemoglobin = st.number_input('Hemoglobin', min_value=0.0, step=0.1)
        packed_cell_volume = st.number_input('Packed Cell Volume', min_value=0)
        white_blood_cell_count = st.number_input('White Blood Cell Count', min_value=0)
        red_blood_cell_count = st.number_input('Red Blood Cell Count', min_value=0.0, step=0.1)

    with col4:
        hypertension = st.number_input('Hypertension', min_value=0)
        diabetes_mellitus = st.number_input('Diabetes Mellitus', min_value=0)
        coronary_artery_disease = st.number_input('Coronary Artery Disease', min_value=0)
        appetite = st.number_input('Appetite', min_value=0)
        pedal_edema = st.number_input('Pedal Edema', min_value=0)
        anemia = st.number_input('Anemia', min_value=0)

    input_features = [[
        age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells,
        pus_cell, pus_cell_clumps, bacteria, blood_glucose_random, blood_urea,
        serum_creatinine, sodium, potassium, hemoglobin, packed_cell_volume,
        white_blood_cell_count, red_blood_cell_count, hypertension,
        diabetes_mellitus, coronary_artery_disease, appetite, pedal_edema, anemia
    ]]
    
    prediction = ""
    if st.button('Prediction'):
        ans = model.predict(input_features)
        prediction = ans[0]
        if prediction == 1:
            st.write('Patient has Kidney Disease.')
        else:
            st.write('Patient does not have Kidney Disease.')

if selected == 'HeartPrediction':
    st.title('Heart Prediction using ML')
