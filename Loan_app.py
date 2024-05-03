import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Initialize an empty list to store submitted data
user_data = []


def user_input_features():
    name = st.text_input('Name')
    gender = st.radio("Gender", options=["Male", "Female", "Other"])
    ApplicantIncome = st.slider("Applicant Income", min_value=150, max_value=10500, value=5000)
    CoapplicantIncome = st.slider("CoapplicantIncome", min_value=150, max_value=6000, value=500)
    LoanAmount = st.slider("LoanAmount", min_value=0, max_value=500, value=250)
    Loan_Amount_Term = st.slider("Loan_Amount_Term", min_value=0, max_value=500, value=250)

    submitted = st.button("Submit")

    if submitted:
        if not (name and gender and ApplicantIncome and CoapplicantIncome and LoanAmount and Loan_Amount_Term):
            st.warning("Please fill all fields before submitting.")
            return None, None
        else:
            # Append the submitted data to the submission history list
            submission_data = {
                "Name": name,
                "Gender": gender,
                "Applicant Income": ApplicantIncome,
                "CoapplicantIncome": CoapplicantIncome,
                "LoanAmount": LoanAmount,
                "Loan_Amount_Term": Loan_Amount_Term
            }
            user_data.append(submission_data)
            data = {
                "ApplicantIncome": ApplicantIncome,
                "CoapplicantIncome": CoapplicantIncome,
                "LoanAmount": LoanAmount,
                "Loan_Amount_Term": Loan_Amount_Term
            }
            features = pd.DataFrame(data, index=[0])
            return features, submission_data
    else:
        # Return None if the form is not submitted
        return None, None


df, submission_data = user_input_features()

if df is not None:
    # st.subheader('User Input Parameters')
    # st.write(df)

    Loan = pd.read_csv("Loan_CData.csv")
    X = Loan[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]]
    y = Loan["Loan_Status"]

    clf = RandomForestClassifier()
    clf.fit(X, y)
    prediction = clf.predict(df)
    prediction_prob = clf.predict_proba(df)

    # st.subheader('Prediction')
    if prediction[0] == 'Y':
        prediction_result = 'Approved'
    else:
        prediction_result = 'Not Approved'
    # st.write(prediction_result)

    st.subheader('Prediction Probability')
    # Create column headings dictionary
    col_headings = {'N': 'No', 'Y': 'Yes'}
    # Rename columns using the dictionary
    prediction_prob_df = pd.DataFrame(prediction_prob, columns=[col_headings[col] for col in clf.classes_])
    st.dataframe(prediction_prob_df.style.format("{:.1%}"), width=200)

    # Add prediction result to submission history
    if submission_data:
        submission_data['Prediction'] = prediction_result

# Display submission history table
st.subheader('Loan Approval Status')
if user_data:
    submission_df = pd.DataFrame(user_data)
    st.table(submission_df)
else:
    st.write("No submissions yet.")
