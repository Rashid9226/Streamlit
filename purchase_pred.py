import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model
# with open('purchase.pkl', 'rb') as handle:
#     model = pickle.load(handle)

from joblib import load

# Load the model
model = load('C:\\Users\\User\\Desktop\\DATA\\pycharm\\Purchase Prediction\\purchase.joblib')



# Function to predict using the loaded model
def predict_purchase(age, salary):
    # Features are age and salary
    features = np.array([age, salary]).reshape(1, -1)  # Reshape for single prediction
    prediction = model.predict(features)
    return prediction[0]


# Streamlit UI
def main():
    st.sidebar.title('Pages')

    # Styling the sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            color: #333;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content .stButton {
            background-color: #6c757d;
            color: white;
            border-radius: 5px;
            margin-bottom: 10px;
            cursor: pointer;
        }
        .sidebar .sidebar-content .stButton:hover {
            background-color: #495057;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    prediction_button = st.sidebar.button('Prediction Page')
    history_button = st.sidebar.button('Purchasing History Data')
    about_button = st.sidebar.button('About')

    # Set default button state
    prediction_selected = True
    history_selected = False
    about_selected = False

    if prediction_button:
        prediction_selected = True
        history_selected = False
        about_selected = False

    if history_button:
        prediction_selected = False
        history_selected = True
        about_selected = False

    if about_button:
        prediction_selected = False
        history_selected = False
        about_selected = True

    if prediction_selected:
        st.title('Customer Purchase Prediction')
        st.write('''
        This project is about classifying Social Media ads means analysing the social media ads for finding the most profitable customers for the product who shows more 
        interest in buying the product. Analysing helps to identify who or which group of people seems to like the product more because as we know how different is our 
        taste from others.
        ''')
        st.write('## Prediction Page')
        st.write('''
        This app predicts whether a customer will make a purchase or not based on the historical purchase data by considering the age and the Salary of the target audience
        ''')

        # User input for features
        name = st.text_input('Full Name')
        age = st.number_input('Age', min_value=0, max_value=150, step=1)
        salary = st.number_input('Estimated Salary', min_value=0, step=1000)

        if st.button('Predict'):
            if not name or age is None or salary is None:  # Check if any field is empty
                st.warning('Please fill in all the fields to make a prediction.')
            else:
                # Make prediction
                prediction = predict_purchase(age, salary)

                # Display prediction
                if prediction == 0:
                    prediction_text = 'Will Not Purchase'
                    st.error(prediction_text)  # Highlight in red for 'Will Not Purchase'
                else:
                    prediction_text = 'Will Purchase'
                    st.success(prediction_text)  # Highlight in green for 'Will Purchase'

    elif history_selected:
        st.title('Purchasing History Data')
        st.write(
            '##### The dataset contains a product’s social media advertising campaign was used for training model which is displayed below:')
        # st.write('The dataset used for training is displayed below:')

        # Load the dataset into a DataFrame
        dataset_url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv'  # Replace with your dataset URL
        dataset = pd.read_csv(dataset_url)

        # Center-align values in the table
        dataset_styled = dataset.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
            'selector': 'td',
            'props': [('text-align', 'center')]
        }])

        # Display the dataset table with centered alignment
        st.table(dataset_styled)

    elif about_selected:
        # st.title('About')
        st.write('## About the Project')
        st.write('''
        The classification of social media ads is all about analyzing the ads for classifying whether your target audience will buy the product or not. It’s a great use case for data science in marketing
        So this is how you can analyze and classify social media ads about the marketing campaign of a product. Classifying social media ads means analyzing your social media ads for finding the most 
        profitable customers for your product who are more likely to buy the product. I hope you liked this article on classifying Social Media Ads with Machine Learning using Python. Feel free to connect with me. 
        ''')

        st.write('## Contact Information')
        st.write('LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/muhammed-rashid01)')
        st.write('GitHub: [GitHub Profile](https://github.com/Rashid9226)')
        st.write('Portfolio: [Novypro Profile](https://www.novypro.com/profile_projects/rashid)')
        st.write('Email: muhammedrashid7274@gmail.com')


if __name__ == '__main__':
    main()
