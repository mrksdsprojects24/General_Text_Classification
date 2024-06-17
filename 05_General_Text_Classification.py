import pandas as pd
from transformers import pipeline
import streamlit as st

st.title("Krishna's General Text Classification App")

text = st.text_input("Enter the text you want to classify:")
candidate_labels = st.text_input("Enter candidate labels (comma-separated):", "")

if st.button("Classify"):
  if text and candidate_labels:
    zsc_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Split candidate labels into a list
    candidate_labels_list = candidate_labels.split(",")
    candidate_labels_list = [label.strip() for label in candidate_labels_list]  # Remove leading/trailing whitespaces

    results = zsc_pipeline(text, candidate_labels=candidate_labels_list)
    df = pd.DataFrame({'labels': results['labels'], 'scores': results['scores']})
    st.write(df)
  else:
    st.warning("Please enter both text and candidate labels.")
