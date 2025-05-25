import streamlit as st
import pickle

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == 0:
            st.error("🚨 FAKE NEWS!")
        else:
            st.success("✅ REAL NEWS!")
