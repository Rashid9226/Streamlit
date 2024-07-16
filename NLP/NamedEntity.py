import streamlit as st
import spacy
from spacy import displacy

nlp=spacy.load('en_core_web_lg')

st.header('Named Entity Recognition')
user_input=st.text_area('Enter Text')

if st.button('Submit'):
    if user_input:
        doc=nlp(user_input)
        html=displacy.render(doc,style='ent',jupyter=False)
        st.write(html,unsafe_allow_html=True)
    else:
        st.write('No Entities Found')
