import streamlit as st
import joblib
from src.preprocess import clean_text
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="AI Review Validator", layout="wide")

# 1. Load Model
@st.cache_resource
def load_model():
    # Using joblib as it is more stable for Random Forest
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

if analyze_btn:
    if review:
        cleaned = clean_text(review)
        
        # 1. Get raw probabilities
        probs = c.predict_proba([review])[0]
        
        # 2. Map them to labels
        # Assuming index 0 = Fake (CG) and index 1 = Real (OR)
        # We find which index has the highest probability
        import numpy as np
        prediction_index = np.argmax(probs)
        
        # 3. Repetition Check (Our safety net)
        words = cleaned.split()
        unique_ratio = len(set(words)) / len(words) if len(words) > 0 else 1
        
        # --- THE FINAL VERDICT LOGIC ---
        # If max probability is at index 0 OR unique_ratio is low -> FAKE
        is_fake = (prediction_index == 0) or (unique_ratio < 0.45 and len(words) > 10)

        st.divider()
        if is_fake:
            st.error("### 🚩 VERDICT: FAKE")
            st.write(f"**Analysis:** The system detected machine-generated patterns. (Confidence: {probs[0]*100:.1f}%)")
        else:
            st.success("### ✅ VERDICT: REAL")
            st.write(f"**Analysis:** This review follows natural human language patterns. (Confidence: {probs[1]*100:.1f}%)")

        # --- LIME SECTION ---
        st.subheader("Visual Explanation")
        map_names = ['Fake (CG)', 'Real (OR)'] 
        explainer = LimeTextExplainer(class_names=map_names)
        
        with st.spinner("Aligning chart..."):
            exp = explainer.explain_instance(review, c.predict_proba, num_features=10)
            lime_html = exp.as_html()
            custom_css = "<style>.lime { color: white !important; } text { fill: white !important; }</style>"
            components.html(custom_css + lime_html, height=600, scrolling=True)
    else:
        st.warning("Please enter a review first!")))

        # HYBRID LOGIC: Check for repetition (Lexical Diversity)
        words = cleaned.split()
        unique_ratio = len(set(words)) / len(words) if len(words) > 0 else 1
        
        # We force FAKE if model says 'CG' OR if repetition is extremely high
        is_fake = (str(prediction).upper() == "CG") or (unique_ratio < 0.45 and len(words) > 10)

        st.divider()
        
        # --- VERDICT DISPLAY (The Final Fix) ---
        st.divider()
        
        # Standardize the prediction to uppercase for comparison
        pred_upper = str(prediction).upper()
        
        # A review is fake if the Model says 'CG' OR if our Repetition Rule triggers
        if pred_upper == "CG" or (unique_ratio < 0.45 and len(words) > 10):
            st.error("### 🚩 VERDICT: FAKE")
            st.write(f"**Analysis:** The system detected bot-like patterns or machine-generated structures.")
            st.write(f"**Uniqueness Score:** {unique_ratio:.2f} (Lower = more repetitive)")
        else:
            st.success("### ✅ VERDICT: REAL")
            st.write(f"**Analysis:** The review appears to be written by a human with natural language variety.")
            st.write(f"**Confidence:** {max(probs)*100:.1f}%")

        # --- LIME SECTION ---
        st.subheader("Visual Explanation")
        
        # Alphabetical alignment: 0=CG (Fake), 1=OR (Real)
        map_names = ['Real (OR)', 'Fake (CG)'] 
        explainer = LimeTextExplainer(class_names=map_names)
        
        with st.spinner("Generating feature importance..."):
            exp = explainer.explain_instance(
                review, 
                c.predict_proba, 
                num_features=10
            )
            
            # CSS for visibility
            lime_html = exp.as_html()
            custom_css = """
            <style>
                .lime { color: white !important; }
                text { fill: white !important; font-family: sans-serif; }
                .lime.label { color: #ffaa00 !important; font-weight: bold; }
                body { background-color: #0e1117; }
            </style>
            """
            components.html(custom_css + lime_html, height=600, scrolling=True)
    else:
        st.warning("Please enter a review first!")
