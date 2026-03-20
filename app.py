import streamlit as st
import pickle
from src.preprocess import clean_text
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🔍 Fake Review Detection + Explainability")

# 1. Load your model and vectorizer
with open("model/fake_review_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# 2. Create a Pipeline for LIME
# This ensures LIME handles the text -> vector -> prediction flow correctly
c = make_pipeline(vectorizer, model)

review = st.text_area("Enter Review to Analyze", height=150)

if st.button("Analyze Review"):
    if review:
        # --- PREDICTION LOGIC ---
        cleaned = clean_text(review)
        prediction = model.predict(vectorizer.transform([cleaned]))[0]
        
        # Mapping: CG (Computer Generated) = FAKE, OR (Original) = REAL
        if prediction == "CG":
            label = "FAKE"
            st.error(f"Prediction: {label}")
        else:
            label = "REAL"
            st.success(f"Prediction: {label}")

        # --- LIME EXPLAINABILITY SECTION ---
        st.subheader("Why did the AI choose this?")
        
        with st.spinner("Calculating word importance..."):
            # Use CG/OR labels for LIME as well to keep it consistent
            explainer = LimeTextExplainer(class_names=['OR', 'CG'])
            
            exp = explainer.explain_instance(
                review, 
                c.predict_proba, 
                num_features=10
            )
            
            # --- CSS to fix visibility (Aggressive fix for SVGs) ---
            lime_html = exp.as_html()
            custom_css = """
            <style>
                * { color: white !important; }
                text { fill: white !important; } /* Fixes the SVG chart labels */
                .lime.label { color: #ffaa00 !important; font-weight: bold; }
                body { background-color: #0e1117; }
            </style>
            """
            final_html = custom_css + lime_html
            components.html(final_html, height=800, scrolling=True)
            
    else:
        st.warning("Please enter some text first.")
