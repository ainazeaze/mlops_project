import streamlit as st
import requests
import os 
import pandas as pd

st.title("Sentiment Analyzer")

webapp_url = os.environ.get('WEBAPP_URL', None)
if webapp_url is None:
    st.error("WEBAPP_URL not defined")
    st.stop()

tab1, tab2 = st.tabs(["🔮 Prediction", "📜 History"])

with tab1:
    st.header("Make a Prediction")
    text = st.text_area("Write your review ...", height=150)

    if st.button("Predict sentiment"):
        if not text:
            st.warning("Please enter some text.")
        else:
            try:
                pred = requests.post(f"http://{webapp_url}/predict", json={"reviews" : [text]})
                
                if pred.status_code == 200:
                    data = pred.json()
                    st.success(f"Sentiment: {data['sentiments']}") 
                else:
                    st.error(f"Error: {pred.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab2:
    st.header("Request History")
    
    n_logs = st.slider("Number of logs to retrieve", min_value=1, max_value=50, value=5)
    
    if st.button("Load History"):
        try:
            history_response = requests.get(f"http://{webapp_url}/history", params={"n": n_logs})
            
            if history_response.status_code == 200:
                history_data = history_response.json()
                
                if history_data:
                    df = pd.DataFrame(history_data)
                    st.dataframe(df)
                else:
                    st.info("No history found in database.")
            else:
                st.error(f"Error retrieving history: {history_response.status_code}")
                
        except Exception as e:
            st.error(f"Connection failed: {e}")