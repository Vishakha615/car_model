import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Car Price Prediction App")

pipe = pickle.load(open("pipe.pkl", "rb+"))
df = pd.read_csv("final_data.csv")
companies = sorted(df["company"].unique())
years = range(2000, 2027)

company = st.sidebar.selectbox("Select company", companies)

names = sorted(df[df['company'] == company]["name"].unique())

name = st.sidebar.selectbox("Select name", names)
year = st.sidebar.selectbox("Select year", years)
km_driven = st.sidebar.number_input("Enter km driven", value=50000, min_value=1000, max_value=200000, step=1000)
fuel = st.sidebar.selectbox("Select fuel type", ["Petrol", "Diesel"])

if st.sidebar.button("Predict Price"):
    st.write("You have selected:")
    st.write(f"Company: {company}") 
    st.write(f"Name: {name}")
    st.write(f"Year: {year}")
    st.write(f"Kilometers Driven: {km_driven}")
    st.write(f"Fuel Type: {fuel}")
    #check for user input
    myinput = [[company, name, year, km_driven, fuel]]
    columns = ['company', 'name', 'year', 'kms_driven', 'fuel_type']
    myinput = pd.DataFrame(data = myinput, columns = columns)
    result = pipe.predict(myinput)

    if result[0,0] < 0:
        st.write("Sorry, the predicted price is negative. Please check your input values.")
    else:
        st.write("Predicted price is:", str(round(result[0,0])))