import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


def user_input_features():
    with st.form("penguin_form"):
        ApplicantIncome=st.slider("Applicant Income",min_value=150,max_value=10500,value=5000)
        CoapplicantIncome = st.slider("CoapplicantIncome", min_value=150, max_value=6000, value=500)
        LoanAmount = st.slider("LoanAmount", min_value=0, max_value=500, value=250)
        Loan_Amount_Term = st.slider("Loan_Amount_Term", min_value=0, max_value=500, value=250)
        submitted = st.form_submit_button("Submit")

    if submitted:
        data = {
            "ApplicantIncome" : ApplicantIncome,
            "CoapplicantIncome" : CoapplicantIncome,
            "LoanAmount" : LoanAmount,
            "Loan_Amount_Term" : Loan_Amount_Term
        }
        features = pd.DataFrame(data,index = [0])
        return features
    else:
        # Return a default DataFrame with all zeros if the form is not submitted
        return pd.DataFrame({
            "ApplicantIncome": [0],
            "CoapplicantIncome": [0],
            "LoanAmount": [0],
            "Loan_Amount_Term": [0]
        })

df=user_input_features()

st.subheader('User Input Parameters')
st.write(df)


Loan = pd.read_csv("Loan_CData.csv")
X = Loan[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]]
y = Loan["Loan_Status"]

clf = RandomForestClassifier()
clf.fit(X,y)
prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_prob)
