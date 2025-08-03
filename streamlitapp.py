import streamlit as st
import predict

# Page Configuration
st.set_page_config(page_title="News Predictor", page_icon="📰", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
        margin-bottom: 20px;
    }
    .subtext {
        text-align: center;
        font-size: 18px;
        color: #7F8C8D;
        margin-bottom: 30px;
    }
    .textbox {
        border: 2px solid #2980B9;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="main-title">📰 Predict News Authenticity</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Paste a news article below and find out if it is real or fake.</div>', unsafe_allow_html=True)

# Input Box inside a card
with st.container():
    text = st.text_area("", height=250, placeholder="Enter your news content...", key="news_input")

# Button and Result
if st.button("🔍 Analyze", use_container_width=True):
    with st.spinner("Analyzing the news..."):
        result = predict.predict(text)
        if result.lower() == "fake":
            st.error("🚨 This news appears to be **FAKE**.")
        elif result.lower() == "real":
            st.success("✅ This news appears to be **REAL**.")
        else:
            st.info(f"🔎 Prediction: **{result}**")
