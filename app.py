import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import pandas as pd
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SOTA Legal Classifier v2.0", page_icon="⚖️", layout="centered")

# --- 2. ADVANCED TEXT PREPROCESSING ---
# CRITICAL: We must format the input exactly how we trained the V2 model
def extract_head_tail(text, head_words=128, tail_words=384):
    words = text.split()
    if len(words) <= 512:
        return text
    return " ".join(words[:head_words] + ["... [TEXT TRUNCATED] ..."] + words[-tail_words:])

# --- 3. LOAD SYSTEM (Cached for speed) ---
@st.cache_resource
def load_system():
    model_path = './legal_bert_v2' 
    
    if not os.path.exists(model_path):
        st.error(f"Cannot find the folder '{model_path}'. Make sure it is extracted in the same folder as app.py!")
        st.stop()

    # Load Tokenizer, Model, and Label Classes
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classes = joblib.load(f'{model_path}/label_classes.joblib')
    
    model.eval()
    return tokenizer, model, classes

with st.spinner("Spinning up V2.0 Legal-BERT Engine..."):
    tokenizer, model, classes = load_system()

# --- 4. INFERENCE PIPELINE ---
def classify_text(raw_text):
    # 1. Apply the Head+Tail extraction first
    processed_text = extract_head_tail(raw_text)
    
    # 2. Tokenize
    inputs = tokenizer(processed_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    
    # 3. Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_label = classes[predicted_idx.item()]
        
    return predicted_label, confidence.item(), probabilities.numpy()

# --- 5. FRONTEND DASHBOARD ---
st.title("⚖️ Legal Outcome Classifier (v2.0)")
st.markdown("Powered by a custom-weighted **Legal-BERT** model trained on 24,000+ precedents. Paste a case below for semantic analysis.")

user_input = st.text_area("Enter Legal Document:", height=250, placeholder="Paste the case facts, procedural history, and holding here...")

if st.button("Classify Document", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please paste some legal text first.")
    else:
        with st.spinner("Analyzing semantics and legal reasoning..."):
            prediction, confidence, probs = classify_text(user_input)
            
        # Display the Winner
        st.success(f"### Predicted Outcome: **{prediction.upper()}**")
        st.info(f"**System Confidence:** {confidence:.2%}")
        
        # Display the Probability Breakdown
        st.subheader("Probability Distribution")
        
        # Format data for the bar chart
        prob_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)
        
        st.bar_chart(data=prob_df, x="Outcome", y="Probability")