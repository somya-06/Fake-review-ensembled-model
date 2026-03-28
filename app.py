from scraper_test import scrape_amazon_reviews, scrape_flipkart_reviews
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

# --- REUSABLE ANALYSIS FUNCTION ---
def run_analysis(review_text):
    cleaned = clean_text(review_text)
    words = cleaned.split()
    
    if len(words) == 0:
        st.warning("Please enter a valid review.")
        return

    probs = c.predict_proba([cleaned])[0]
    prediction_index = np.argmax(probs)
    ai_confidence = probs[1] * 100
    
    unique_ratio = len(set(words)) / len(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0

    is_fake = (prediction_index == 0) or (unique_ratio < 0.15) or (avg_word_length > 10)

    with st.expander("📊 Technical Analysis"):
        col1, col2, col3 = st.columns(3)
        col1.metric("AI Real Confidence", f"{ai_confidence:.1f}%")
        col2.metric("Uniqueness Score", f"{unique_ratio:.2f}")
        col3.metric("Avg Word Length", f"{avg_word_length:.1f}")

    if is_fake:
        st.error("### 🚩 VERDICT: FAKE")
    else:
        st.success("### ✅ VERDICT: REAL")

    st.subheader("🔍 Visual Explanation")
    explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
    exp = explainer.explain_instance(cleaned, c.predict_proba, num_features=10)
    components.html(exp.as_html(), height=450, scrolling=True)

# --- UI LAYOUT TABS ---
tab1, tab2 = st.tabs(["📝 Manual Input", "🌐 Live E-Commerce Scraper"])

with tab1:
    st.subheader("Analyze a Single Review")
    manual_review = st.text_area("Paste review here:", height=150, key="manual_area")
    if st.button("Analyze", key="manual_btn"):
        if manual_review:
            run_analysis(manual_review)

# --- TAB 2: UNIVERSAL SCRAPER ---
with tab2:
    st.subheader("🌐 Universal Product Review Analysis")
    product_url = st.text_input("Paste Amazon or Flipkart URL:", key="scraper_url_input")

    if st.button("Extract & Analyze Reviews", key="url_btn"):
        if product_url:
            with st.spinner("Detecting site and fetching reviews..."):
                # --- SITE DETECTION LOGIC ---
                if "flipkart.com" in product_url or "fkrt.it" in product_url:
                    reviews = scrape_flipkart_reviews(product_url)
                    site_name = "Flipkart"
                elif "amazon" in product_url or "amzn.in" in product_url:
                    reviews = scrape_amazon_reviews(product_url)
                    site_name = "Amazon"
                else:
                    st.error("Platform not supported. Please use Amazon or Flipkart.")
                    reviews = None

            if reviews:
                real_count = 0
                total_reviews = len(reviews)
                
                for r_text in reviews:
                    cleaned = clean_text(r_text)
                    probs = c.predict_proba([cleaned])[0]
                    if np.argmax(probs) == 1: real_count += 1
                
                # --- INTEGRITY SCORE CALCULATIONS ---
                ai_star_rating = (real_count / total_reviews) * 5
                
                st.divider()
                st.header(f"🛡️ {site_name} Integrity Report")
                
                col_stars, col_metrics = st.columns([1, 2])
                with col_stars:
                    st.metric("Overall AI Trust Rating", f"{ai_star_rating:.1f} / 5")
                    stars = "⭐" * int(round(ai_star_rating))
                    st.subheader(f"{stars if stars else '🌑'}")

                with col_metrics:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Reviews", total_reviews)
                    c2.metric("Real Found", real_count)
                    c3.metric("Fakes Flagged", total_reviews - real_count)

                if ai_star_rating >= 4.0:
                    st.success("### ✅ VERDICT: HIGH INTEGRITY")
                elif ai_star_rating >= 2.5:
                    st.warning("### ⚠️ VERDICT: MIXED SIGNALS")
                else:
                    st.error("### 🚫 VERDICT: UNTRUSTWORTHY")

                st.divider()
                st.subheader("📑 Review Breakdown")
                for i, r_text in enumerate(reviews):
                    with st.expander(f"Review {i+1}"):
                        st.write(r_text)
                        run_analysis(r_text)
