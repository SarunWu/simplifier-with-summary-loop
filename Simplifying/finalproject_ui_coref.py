import streamlit as st
import simplifier
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

import sys

sys.path.insert(0, '..')
from coreference_resolution.generate_summary import generate_coreference_summary as coref

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = ["loaded"]
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/pegasus-cnn_dailymail"

    print("Downloading model")
    st.session_state.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(st.session_state.device)
    st.session_state.tokenizer = PegasusTokenizer.from_pretrained(model_name)

    st.session_state.rougescore = 0

st.header("NLP Final Project")

col1, col2, col3 = st.columns(3)

with col1:
    number_candidates = st.number_input('Beam search number', value=2, max_value=5, min_value=1, format="%d")
    number_candidates = int(number_candidates)

with col2:
    max_length = st.number_input('Max length', value=50, min_value=10, format="%d")
    max_length = int(max_length)

with col3:
    ""

# 1 - Input text for summarization
# Summarized Text
with st.container():
    input_text = st.text_area('Introduce your text to summarize!', value="", height=100)
    st.text("Text length: " + str(len(input_text)))

st.markdown("""---""")

# 2 - Generate co-references
# extractive_summarize_button = st.button("Generate extractive summary")
used_pronoun = st.checkbox("Replace pronoun")

original_summary = ""
coref_summary = ""

with st.container():
    extracted_summary = ""
    if input_text != "":
        original_summary, coref_summary = coref.generate_summary(input_text)
        extracted_summary = original_summary
        if used_pronoun:
            extracted_summary = coref_summary
    st.text_area('Extractive Summary', extracted_summary, height=150)
    st.text("Text length: " + str(len(extracted_summary)))

# 3 - Generate Summary
# summarize_button = st.button("Generate abstractive summary")

st.markdown("""---""")

with st.container():
    used_coref = st.checkbox("Use extractive summary as input")
    if ~used_coref:
        extracted_summary = input_text

    post_sum = ""
    if input_text != "":
        inputs = st.session_state.tokenizer(extracted_summary, max_length=1024, return_tensors="pt",
                                            truncation=True).to(
            st.session_state.device)
        summary_ids = st.session_state.model.generate(inputs["input_ids"], num_beams=number_candidates,
                                                      max_length=max_length)
        post_sum = \
            st.session_state.tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)[0]
        post_sum = post_sum.replace("<n>", " ")
        difficult_words = simplifier.diffword(post_sum)

    st.text_area('Summary Text', post_sum, height=150)
    st.text("Text length: " + str(len(post_sum)))

st.markdown("""---""")

# 4 - Simplified Text

# simplify_button = st.button("Simplify the summary")
with st.container():
    simplified_coref = st.checkbox("Simplify extractive summary")
    simplified_input = post_sum
    if simplified_coref:
        simplified_input = coref_summary

    simplified_summary = simplifier.printsim(simplifier.simplify(simplified_input))

    st.text_area('Simplified Text', simplified_summary, height=100)
    st.text("Text length: " + str(len(simplified_summary)))
