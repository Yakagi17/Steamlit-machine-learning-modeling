import streamlit as st
import modeling_demo
from modeling.temp import get_data
from streamlit.logger import get_logger
from collections import OrderedDict



LOGGER = get_logger(__name__)

MODELING_DEMO = OrderedDict(
    [
        ("---",(modeling_demo.startup, None)),
        ("Classification",(modeling_demo.classification_modeling, "Desc")),
        ("Regression",(modeling_demo.regression_modeling, "Desc")),
        ("Clustering",(modeling_demo.clustering_modeling, "Desc")),
    ])

def run():
    modeling_demo_name = st.sidebar.selectbox("Choose Modeling Type :", list(MODELING_DEMO.keys()),0)
    demo_app = MODELING_DEMO[modeling_demo_name][0]

    if modeling_demo_name == "---":
        st.write("# Welcome to Machine Learning - Modeling")
    else:
        st.markdown(f'# {modeling_demo_name}')
        modeling_description = MODELING_DEMO[modeling_demo_name][1]
        if modeling_description:
            st.write(modeling_description)

    get_data()    
        # for i in range(10):
        #     st.empty

    demo_app()

if __name__ == "__main__":
    run()
