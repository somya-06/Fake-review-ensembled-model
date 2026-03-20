🛡️ AI Review Integrity System
An AIML-based system designed to detect Machine-Generated (CG) vs. Original (OR) reviews using **Random Forest** and **LIME** (Local Interpretable Model-agnostic Explanations).

##  Features
* **Hybrid Detection:** Combines ML predictions with Heuristic Lexical Analysis (Uniqueness & Generic Word Density).
* **Explainable AI (XAI):** Integrated LIME charts to show *why* a review was flagged.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Framework:** Scikit-Learn (Random Forest + TF-IDF)
* **Explanation:** LIME
* **UI:** Streamlit

## 📊 Logic Metrics
The system evaluates reviews based on:
1.  **ML Confidence:** Raw probability from the Random Forest model.
2.  **Lexical Diversity:** Ratio of unique words to total words.
3.  **Generic Density:** Percentage of "filler" marketing praise words.
