import streamlit as st
import modeling_demo
from modeling.temp import get_dataset
from streamlit.logger import get_logger
from collections import OrderedDict



LOGGER = get_logger(__name__)

dataset_column = []


MODELING_DEMO = OrderedDict(
    [
        ("---", modeling_demo.startup),
        ("Classification", modeling_demo.classification_modeling),
        ("Regression", modeling_demo.regression_modeling),
        ("Clustering", modeling_demo.clustering_modeling),
    ])


#Main Page (Right)
def ml_demo_interface():
    modeling_demo_name = st.sidebar.selectbox("Choose Modeling Type :", list(MODELING_DEMO.keys()),0)
    demo_app = MODELING_DEMO[modeling_demo_name]
    if modeling_demo_name == "---":
        st.write("# Welcome to Machine Learning - Modeling")
    else:
        st.markdown(f'# {modeling_demo_name}')

    dataset_df, dataset_column = get_dataset()    

    if modeling_demo_name != "---" and not dataset_df.empty:
        demo_app(dataset_df, dataset_column)
    elif modeling_demo_name != "---" and dataset_df.empty:
        st.sidebar.error("Please upload data set first")
    else:
        demo_app()


if __name__ == "__main__":
    ml_demo_interface()
