import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Car Price Prediction App")

pipe = pickle.load(open("pipe.pkl", "rb"))

df = pd.read_csv("Cleaned_data.csv")

df.columns = df.columns.str.strip().str.lower()

required = ['company', 'name', 'year', 'kms_driven', 'fuel_type']

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

companies = sorted(df["company"].dropna().unique())
years = range(2000, 2027)

company = st.sidebar.selectbox("Select company", companies)

names = sorted(df[df["company"] == company]["name"].unique())

name = st.sidebar.selectbox("Select name", names)
year = st.sidebar.selectbox("Select year", years)
km_driven = st.sidebar.number_input(
    "Enter km driven",
    value=50000,
    min_value=1000,
    max_value=200000,
    step=1000
)

fuel = st.sidebar.selectbox("Select fuel type", ["Petrol", "Diesel"])

if st.sidebar.button("Predict Price"):

    myinput = pd.DataFrame(
        [[company, name, year, km_driven, fuel]],
        columns=['company', 'name', 'year', 'kms_driven', 'fuel_type']
    )

    result = pipe.predict(myinput)

    st.write("Predicted price:", round(result[0]))
