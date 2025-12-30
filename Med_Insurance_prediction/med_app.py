import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  

st.title('MEDICAL INSURANCE')

model = load_model('med_ins_ann.keras')
model.compile(optimizer='adam',
              loss = 'mean_squared_error',
              metrics=['mean_squared_error'])



model = tf.keras.models.load_model("med_ins_ann.h5")


age = st.number_input('Age')  #18 - 64
gender = st.selectbox('Gender',['Male','Female'])
BMI = st.number_input('BMI')   # 16 - 53 
Children = st.selectbox('Children',[0,1,2,3,4,5])
smoker = st.selectbox('Smoker',['yes','no'])
region = st.selectbox('Region',['southwest', 'southeast', 'northwest', 'northeast'])


data = pd.DataFrame({'age': [age],'sex': [gender],'bmi': [BMI],'children':[Children],'smoker':[smoker],'region':[region]})

gender = 1 if gender == "Male" else 0
smoker = 1 if smoker == "yes" else 0
region_southwest = 1 if region == 'southwest' else 0
region_southeast = 1 if region == 'southeast' else 0
region_northwest = 1 if region == 'northwest' else 0
region_northeast = 1 if region == 'northeast' else 0

# if st.button("Predict"):
#     prediction = model.predict([[age,gender, BMI,Children, smoker, region_southwest,
#     region_southeast,
#     region_northwest,
#     region_northeast]])
#     st.success(f"Prediction: {prediction[0]}")


if st.button("Predict"):
    input_data = np.array([[
        age,
        gender,
        BMI,
        Children,
        smoker,
        region_southwest,
        region_southeast,
        region_northwest,
    ]], dtype=np.float32)

    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0][0]}")
