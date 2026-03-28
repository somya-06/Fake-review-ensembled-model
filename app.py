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

    # 1. Prediction
    probs = c.predict_proba([cleaned])[0]
    prediction_index = np.argmax(probs)
    ai_confidence = probs[1] * 100
    
    # 2. Heuristics
    unique_ratio = len(set(words)) / len(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    is_fake = (prediction_index == 0) or (unique_ratio < 0.15) or (avg_word_length > 10)

    # --- UI DISPLAY ---
    col_v, col_t = st.columns([1, 2])
    
    with col_v:
        if is_fake:
            st.error("### 🚩 FAKE")
        else:
            st.success("### ✅ REAL")
        st.caption(f"AI Confidence: {ai_confidence:.1f}%")

    with col_t:
        with st.expander("📊 View Technical Metrics"):
            st.write(f"- **Uniqueness:** {unique_ratio:.2f}")
            st.write(f"- **Avg Word Len:** {avg_word_length:.1f}")
            st.write(f"- **Raw AI Verdict:** {'Fake' if prediction_index == 0 else 'Real'}")

    # LIME Explanation
    st.markdown("---")
    st.write("**🔍 Feature Importance (LIME)**")
    explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
    exp = explainer.explain_instance(cleaned, c.predict_proba, num_features=8)
    components.html(exp.as_html(), height=350, scrolling=False)

# --- UI LAYOUT TABS ---
tab1, tab2 = st.tabs(["📝 Manual Input", "🌐 Live E-Commerce Scraper"])

with tab1:
    st.subheader("Analyze a Single Review")
    manual_review = st.text_area("Paste review here:", height=150, key="manual_area")
    if st.button("Run Analysis", key="manual_btn"):
        if manual_review:
            run_analysis(manual_review, key_suffix="manual")

with tab2:
    st.subheader("🌐 Universal Product Review Analysis")
    product_url = st.text_input("Paste Amazon or Flipkart URL:", key="scraper_url_input")

    if st.button("Extract & Analyze Reviews", key="url_btn"):
        if product_url:
            with st.spinner("Detecting site and fetching reviews..."):
                # SITE DETECTION
                if "flipkart.com" in product_url or "fkrt.it" in product_url:
                    reviews = scrape_flipkart_reviews(product_url)
                    site_name = "Flipkart"
                elif "amazon" in product_url or "amzn.in" in product_url:
                    reviews = scrape_amazon_reviews(product_url)
                    site_name = "Amazon"
                else:
                    st.error("Platform not supported. Use Amazon or Flipkart link.")
                    reviews = None

            if reviews:
                # --- CALCULATIONS ---
                total_reviews = len(reviews)
                real_count = 0
                for r_text in reviews:
                    cleaned = clean_text(r_text)
                    if np.argmax(c.predict_proba([cleaned])[0]) == 1:
                        real_count += 1
                
                ai_star_rating = (real_count / total_reviews) * 5
                
                # --- REPORT HEADER ---
                st.divider()
                st.header(f"🛡️ {site_name} Integrity Report")
                
                c_stars, c_metrics = st.columns([1, 2])
                with c_stars:
                    st.metric("Overall AI Trust", f"{ai_star_rating:.1f} / 5")
                    stars = "⭐" * int(round(ai_star_rating))
                    st.subheader(stars if stars else "🌑")

                with c_metrics:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Analyzed", total_reviews)
                    m2.metric("Real", real_count)
                    m3.metric("Flagged", total_reviews - real_count)

                # --- SUMMARY VERDICT ---
                if ai_star_rating >= 4.0:
                    st.success("### ✅ VERDICT: SAFE PRODUCT")
                elif ai_star_rating >= 2.5:
                    st.warning("### ⚠️ VERDICT: CAUTION ADVISED")
                else:
                    st.error("### 🚫 VERDICT: HIGH RISK / FAKE REVIEWS")

                # --- DETAILED BREAKDOWN ---
                st.divider()
                st.subheader("📑 Detailed Review Breakdown")
                for i, r_text in enumerate(reviews):
                    with st.expander(f"Review {i+1} - Details", expanded=False):
                        st.info(f"**Text:** {r_text[:200]}...")
                        run_analysis(r_text, key_suffix=f"scrape_{i}")
