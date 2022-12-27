import streamlit as st
import pandas as pd
import numpy as np

st.header("NLP Final Project")

# Number of candidates
candidates = ['1', '2', '3']
number_candidates = st.selectbox("Select number of candidate summarization to display.", candidates)

# Input text for summarization
input_text = st.text_input("Introduce your text to summarize!", value="", key="alex")

# Summarized Text
summarized_text = st.text_area('Summarized Text', input_text, height=100)

# Button to apply simplification
clear_button = st.button("Simplify the summary")

# Simplified Text
simplified_text = ""
if clear_button:
    simplified_text = input_text[0:33]
st.text_area('Simplified Text', simplified_text, height=100)
