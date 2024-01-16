import streamlit as st
import pandas as pd

st.title("Welcome to CSC649")
st.header("This is execrise on the implementation of interface development via StreamLit")
st.write("This is the dataset on iris")

read_data = pd.read_csv("../Datasets/turnover.csv", encoding="latin-1")
read_data