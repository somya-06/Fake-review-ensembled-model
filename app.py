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

    # Prediction & Confidence
    probs = c.predict_proba([cleaned])[0]
    prediction_index = np.argmax(probs)
    ai_confidence = probs[1] * 100
    
    # Heuristics
    unique_ratio = len(set(words)) / len(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    is_fake = (prediction_index == 0) or (unique_ratio < 0.15) or (avg_word_length > 10)

    # UI Result Layout
    col_v, col_t = st.columns([1, 2])
    with col_v:
        if is_fake:
            st.error("### 🚩 VERDICT: FAKE")
        else:
            st.success("### ✅ VERDICT: REAL")
        st.caption(f"AI Real Confidence: {ai_confidence:.1f}%")

    with col_t:
        with st.expander("📊 Technical Analysis"):
            st.write(f"- **Uniqueness Score:** {unique_ratio:.2f}")
            st.write(f"- **Avg Word Length:** {avg_word_length:.1f}")

    # LIME Explanation
    st.write("**🔍 Feature Importance (LIME)**")
    explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
    exp = explainer.explain_instance(cleaned, c.predict_proba, num_features=8)
    components.html(exp.as_html(), height=350, scrolling=False)

# --- REUSABLE REPORT GENERATOR FOR SCRAPERS ---
def generate_scraper_report(reviews, site_name):
    if not reviews:
        st.error(f"Could not extract reviews from {site_name}. Check the URL.")
        return

    real_count = 0
    total_reviews = len(reviews)
    
    for r_text in reviews:
        cleaned = clean_text(r_text)
        if np.argmax(c.predict_proba([cleaned])[0]) == 1:
            real_count += 1
    
    ai_star_rating = (real_count / total_reviews) * 5
    
    st.divider()
    st.header(f"🛡️ {site_name} Integrity Report")
    
    col_stars, col_metrics = st.columns([1, 2])
    with col_stars:
        st.metric("Overall AI Trust Rating", f"{ai_star_rating:.1f} / 5")
        stars = "⭐" * int(round(ai_star_rating))
        st.subheader(stars if stars else "🌑")

    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Reviews", total_reviews)
        m2.metric("Real Found", real_count)
        m3.metric("Fakes Flagged", total_reviews - real_count)

    # Overall Verdict
    if ai_star_rating >= 4.0:
        st.success("### ✅ VERDICT: HIGH INTEGRITY PRODUCT")
    elif ai_star_rating >= 2.5:
        st.warning("### ⚠️ VERDICT: MIXED SIGNALS / CAUTION")
    else:
        st.error("### 🚫 VERDICT: UNTRUSTWORTHY / HIGH RISK")

    st.divider()
    st.subheader("📑 Review-by-Review Breakdown")
    for i, r_text in enumerate(reviews):
        with st.expander(f"Review {i+1} Details", expanded=False):
            st.write(f"**Original Text:** {r_text}")
            run_analysis(r_text, key_suffix=f"{site_name}_{i}")

# --- UI LAYOUT TABS ---
tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📦 Amazon Scraper", "🛒 Flipkart Scraper"])

# TAB 1: Manual Check
with tab1:
    st.subheader("Analyze a Single Review")
    manual_review = st.text_area("Paste review text:", height=150, key="manual_area")
    if st.button("Run Analysis", key="manual_btn"):
        if manual_review:
            run_analysis(manual_review, key_suffix="manual")

# TAB 2: Amazon Scraper
with tab2:
    st.subheader("📦 Amazon Product Analysis")
    amz_url = st.text_input("Paste Amazon URL:", key="amz_url_input")
    if st.button("Extract & Analyze Amazon", key="amz_btn"):
        if amz_url:
            with st.spinner("Scraping Amazon reviews..."):
                res = scrape_amazon_reviews(amz_url)
                generate_scraper_report(res, "Amazon")

# TAB 3: Flipkart Scraper
with tab3:
    st.subheader("🛒 Flipkart Product Analysis")
    flp_url = st.text_input("Paste Flipkart URL:", key="flp_url_input")
    if st.button("Extract & Analyze Flipkart", key="flp_btn"):
        if flp_url:
            with st.spinner("Scraping Flipkart reviews..."):
                res = scrape_flipkart_reviews(flp_url)
                generate_scraper_report(res, "Flipkart")
