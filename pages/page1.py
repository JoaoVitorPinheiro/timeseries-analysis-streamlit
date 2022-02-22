import streamlit as st
from dashboard import *
from analysis import *

def create_page1(df, data_group, y_true, y_predicted):
    with st.expander("Dados"):
        st.dataframe(df)
    create_global_metrics(df, data_group, y_true, y_predicted)   
