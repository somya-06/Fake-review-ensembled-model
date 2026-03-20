import streamlit as st
import joblib
import numpy as np
from src.preprocess import clean_text
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="AI Review Validator", layout="wide")

# 1. Load Model
@st.cache_resource
def load_model():
    return joblib.load("model/fake_review_model.pkl")

try:
    model, vectorizer = load_model()
    c = make_pipeline(vectorizer, model)
except Exception as e:
    st.error(f"Model Load Error: {e}. Ensure 'model/fake_review_model.pkl' exists.")

st.title("🛡️ AI Review Integrity System")

# 2. Session State for input
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

def clear_text():
    st.session_state['input_text'] = ""

# Input UI
review = st.text_area("Paste review here:", value=st.session_state['input_text'], height=150, key="review_area")

col1, col2 = st.columns([1, 5])
with col1:
    analyze_btn = st.button("Analyze")
with col2:
    st.button("Clear Text", on_click=clear_text)

# --- LOGIC BLOCK ---
# --- LOGIC BLOCK ---
if analyze_btn:
    if review:
        cleaned = clean_text(review)
        
        # 1. Get raw probabilities
        probs = c.predict_proba([review])[0]
        prediction_index = np.argmax(probs)
        
        # 2. Repetition Check
        words = cleaned.split()
        unique_ratio = len(set(words)) / len(words) if len(words) > 1 else 1
        
        # FINAL VERDICT
        is_fake = (prediction_index == 0) or (unique_ratio < 0.45 and len(words) > 10)

        st.divider()
        if is_fake:
            st.error("### 🚩 VERDICT: FAKE")
            st.write(f"**System Confidence:** {probs[0]*100:.1f}%")
        else:
            st.success("### ✅ VERDICT: REAL")
            st.write(f"**System Confidence:** {probs[1]*100:.1f}%")

        # --- DARK THEME LIME SECTION ---
        st.subheader("Visual Explanation")
        map_names = ['Fake (CG)', 'Real (OR)'] 
        explainer = LimeTextExplainer(class_names=map_names)
        
        with st.spinner("Generating dark-mode visual..."):
            exp = explainer.explain_instance(review, c.predict_proba, num_features=10)
            lime_html = exp.as_html()
            
            # This CSS forces the LIME internal HTML to be Dark Mode compatible
            custom_css = """
            <style>
                /* Force background to match Streamlit's dark theme */
                body { background-color: #0e1117 !important; color: white !important; }
                
                /* Target LIME specific text elements */
                .lime { color: white !important; }
                text { fill: white !important; font-family: sans-serif !important; font-size: 14px !important; }
                
                /* Make the labels stand out */
                .lime.label { color: #ffaa00 !important; font-weight: bold !important; }
                
                /* Ensure the horizontal bars are visible */
                rect { stroke: #444 !important; }
            </style>
            """
            
            # Combine the CSS with the LIME HTML inside the component
            components.html(custom_css + lime_html, height=600, scrolling=True)
    else:
        st.warning("Please enter a review first!")
       
