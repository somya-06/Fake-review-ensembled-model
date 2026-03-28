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
def run_analysis(review_text, key_suffix=""):
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

    col_v, col_t = st.columns([1, 2])
    with col_v:
        if is_fake:
            st.error("### 🚩 FAKE")
        else:
            st.success("### ✅ REAL")
        st.caption(f"AI Confidence: {ai_confidence:.1f}%")

    with col_t:
        with st.expander("📊 Technical Metrics"):
            st.write(f"- **Uniqueness:** {unique_ratio:.2f}")
            st.write(f"- **Avg Word Len:** {avg_word_length:.1f}")

    st.write("**🔍 Feature Importance (LIME)**")
    explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
    exp = explainer.explain_instance(cleaned, c.predict_proba, num_features=8)
    components.html(exp.as_html(), height=350)

# --- UI LAYOUT TABS ---
tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📦 Amazon Scraper", "🛒 Flipkart Scraper"])

# TAB 1: Manual
with tab1:
    st.subheader("Analyze a Single Review")
    manual_review = st.text_area("Paste review text:", height=150, key="manual_area")
    if st.button("Run Analysis", key="manual_btn"):
        if manual_review:
            run_analysis(manual_review, key_suffix="manual")

# --- COMMON SCRAPER UI LOGIC ---
def display_scraper_results(reviews, site_name):
    if reviews:
        total = len(reviews)
        real_count = 0
        for r in reviews:
            if np.argmax(c.predict_proba([clean_text(r)])[0]) == 1:
                real_count += 1
        
        score = (real_count / total) * 5
        st.divider()
        st.header(f"🛡️ {site_name} Integrity Report")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("AI Trust Score", f"{score:.1f} / 5")
            st.subheader("⭐" * int(round(score)) if score > 0 else "🌑")
        with c2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", total); m2.metric("Real", real_count); m3.metric("Fake", total-real_count)

        st.divider()
        st.subheader("📑 Detailed Review Breakdown")
        for i, r_text in enumerate(reviews):
            with st.expander(f"Review {i+1} Details", expanded=False):
                st.write(f"**Review Content:** {r_text}")
                run_analysis(r_text, key_suffix=f"{site_name}_{i}")
    else:
        st.error(f"Could not extract reviews from {site_name}. Check the URL.")

# TAB 2: Amazon
with tab2:
    st.subheader("Amazon Product Analysis")
    amz_url = st.text_input("Paste Amazon URL:", key="amz_url")
    if st.button("Analyze Amazon", key="amz_btn"):
        with st.spinner("Scraping Amazon..."):
            res = scrape_amazon_reviews(amz_url)
            display_scraper_results(res, "Amazon")

# TAB 3: Flipkart
with tab3:
    st.subheader("Flipkart Product Analysis")
    flp_url = st.text_input("Paste Flipkart URL:", key="flp_url")
    if st.button("Analyze Flipkart", key="flp_btn"):
        with st.spinner("Scraping Flipkart..."):
            res = scrape_flipkart_reviews(flp_url)
            display_scraper_results(res, "Flipkart")
