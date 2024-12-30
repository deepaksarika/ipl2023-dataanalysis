import streamlit as st
import pandas as pd
from pandasai.llm import GooglePalm
from pandasai import SmartDataframe
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')
llm = GooglePalm(api_key=api_key)

# Initialize LLM

pandas_ai = PandasAI(llm)

# Streamlit App
st.title("IPL 2023 Data Analysis")

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Display DataFrame shape and first few rows
    st.write("Dataset shape: ", df.shape)
    st.write(df.head())
    
    # Drop 'Unnamed: 0' column
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    # Display the cleaned DataFrame
    st.write("Cleaned DataFrame:")
    st.write(df.head())
    
    # Use PandasAI to analyze the data
    st.subheader("Analysis")
    result = pandas_ai.run(df, prompt="Which players are the most costliest buys?")
    
    # Display the result
    st.write(result)

