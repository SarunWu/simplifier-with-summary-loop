import streamlit as st
import Generate_summary_NLP as simplifier
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = ["loaded"]
    model_name = "google/pegasus-cnn_dailymail"
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Downloading model")
    st.session_state.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(st.session_state.device)
    st.session_state.tokenizer = PegasusTokenizer.from_pretrained(model_name)
    

st.header("NLP Final Project")

col1, col2 = st.columns(2)

with col1:
    number_candidates = st.number_input('Beam search number', value=2, max_value=5, min_value=1, format="%d")
    number_candidates = int(number_candidates)

with col2:
    max_length = st.number_input('Max length', value=50, min_value=10, format="%d")
    max_length = int(max_length)




# Input text for summarization
# input_text = st.text_input("Introduce your text to summarize!", value="", key="alex")
# Summarized Text
input_text = st.text_area('Introduce your text to summarize!', value="", height=100)

# Button to apply simplification
summarize_button = st.button("Generate the summary")

# Generate Summary
post_sum = ""
if input_text != "":
    inputs = st.session_state.tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True).to(st.session_state.device)
    summary_ids = st.session_state.model.generate(inputs["input_ids"], num_beams=number_candidates, max_length=max_length)
    post_sum = st.session_state.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    difficult_words = simplifier.diffword(post_sum)

st.text_area('Summary Text', post_sum, height=150)

# Button to apply simplification
simplify_button = st.button("Simplify the summary")

# Simplified Text
simplified_summary = ""
if simplify_button:
    # Generate SIMPLIFIED Summary
    simplified_summary = simplifier.printsim(simplifier.simplify(post_sum))
st.text_area('Simplified Text', simplified_summary, height=100)
