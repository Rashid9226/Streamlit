import streamlit as st
import pandas as pd
import random

# Upload the file
data = st.file_uploader("Choose the Data File", type="xlsx")

# Check if a file has been uploaded
if data is not None:
    # Read the Excel file
    df = pd.read_excel(data)
    
    st.write(df)
    cols=df.columns
    col=st.selectbox("select Subject: ",df.columns)
    
    if st.button('Random Topic'):
        if not df[col].empty:
            random_topic = random.choice(df[col].dropna().tolist())
            st.success(random_topic)
        else:
            st.warning(f'No topics available in the selected column: {col}')
    
            
    

