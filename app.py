import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import softmax
import plotly.express as px
import sys
import os

# Check if packages are installed
try:
    from transformers import BertForQuestionAnswering, BertTokenizerFast
    import torch
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# Initialize session state
@st.cache_resource
def load_model():
    """Load the BERT model and tokenizer"""
    try:
        model_name = "deepset/bert-base-cased-squad2"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertForQuestionAnswering.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_answer(context, question, tokenizer, model):
    """Predict answer using BERT model"""
    try:
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_scores = softmax(outputs.start_logits.numpy()[0])
        end_scores = softmax(outputs.end_logits.numpy()[0])
        
        start_idx = np.argmax(start_scores)
        end_idx = np.argmax(end_scores)
        
        confidence_score = (start_scores[start_idx] + end_scores[end_idx]) / 2
        
        answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        
        return answer, confidence_score, start_scores, end_scores
    except Exception as e:
        return f"Error: {str(e)}", 0.0, [], []

# Streamlit UI
st.set_page_config(
    page_title="BERT Question Answering",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 BERT Question Answering System")
st.markdown("Ask questions about any context using BERT model!")

# Load model with progress
if 'model_loaded' not in st.session_state:
    with st.spinner('Loading BERT model... This may take a minute.'):
        tokenizer, model = load_model()
        if tokenizer and model:
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model. Please check the requirements.")

# Only show the rest if model is loaded
if st.session_state.get('model_loaded', False):
    # Input sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Context")
        context = st.text_area(
            "Enter your context here:",
            height=200,
            value="""Artificial Intelligence (AI) is the field of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence. These tasks include reasoning, learning, problem-solving, perception, and natural language understanding.

    AI can be classified into two categories: narrow AI, which is designed for specific tasks like speech recognition or image classification, and general AI, which aims to perform any intellectual task that a human can do. Machine learning and deep learning are subfields of AI that have enabled major breakthroughs in computer vision, natural language processing, and robotics."""
        )

    with col2:
        st.subheader("❓ Question")
        question = st.text_input("Enter your question:", "What are the two categories of AI?")

    # Process question
    if st.button("Get Answer", type="primary"):
        if context.strip() and question.strip():
            with st.spinner("Analyzing with BERT..."):
                answer, confidence, start_scores, end_scores = predict_answer(
                    context, question, st.session_state.tokenizer, st.session_state.model
                )
            
            # Display results
            st.subheader("🎯 Answer")
            if answer.startswith("Error:"):
                st.error(answer)
            else:
                st.success(f"**{answer}**")
                st.info(f"**Confidence Score: {confidence:.2%}**")
                
                # Visualization
                if len(start_scores) > 0 and len(end_scores) > 0:
                    st.subheader("📊 Token Confidence Scores")
                    
                    scores_df = pd.DataFrame({
                        "Token Position": list(range(len(start_scores))) * 2,
                        "Score": list(start_scores) + list(end_scores),
                        "Score Type": ["Start"] * len(start_scores) + ["End"] * len(end_scores),
                    })
                    
                    fig = px.bar(
                        scores_df, 
                        x="Token Position", 
                        y="Score", 
                        color="Score Type",
                        barmode="group", 
                        title="Start and End Token Scores",
                        color_discrete_map={"Start": "#636efa", "End": "#EF553B"}
                    )
                    
                    fig.update_layout(
                        xaxis_title="Token Position",
                        yaxis_title="Confidence Score",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Please provide both context and question!")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info(
    """
    **BERT Question Answering System**
    
    Uses `deepset/bert-base-cased-squad2` model fine-tuned on SQuAD 2.0.
    
    **Features:**
    - Extractive question answering
    - Confidence scoring
    - Token-level visualization
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with 🤗 Transformers + Streamlit")
