import streamlit as st
import Generate_summary_NLP as simplifier
import pandas as pd
import numpy as np

st.header("NLP Final Project")

# Number of candidates
candidates = ['1', '2', '3']
number_candidates = st.selectbox("Select number of candidate summarization to display.", candidates)

# Input text for summarization
input_text = st.text_input("Introduce your text to summarize!", value="", key="alex")

# Generate Summary
inputs = simplifier.tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True).to(simplifier.device)
summary_ids = simplifier.model.generate(inputs["input_ids"], num_beams=2, max_length=50)
post_sum = simplifier.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# Summarized Text
summarized_text = st.text_area('Summarized Text', post_sum, height=100)

# Button to apply simplification
clear_button = st.button("Simplify the summary")


# Simplified Text
simplified_summary = ""
if clear_button:
    # Generate SIMPLIFIED Summary
    simplified_summary = simplifier.printsim(simplifier.simplify(post_sum))
st.text_area('Simplified Text', simplified_summary, height=100)


