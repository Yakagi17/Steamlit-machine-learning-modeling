import streamlit as st
from interface.main_interface import classification_modeling, regression_modeling, clustering_modeling, startup
from utils.data_management import get_dataset
from streamlit.logger import get_logger
from collections import OrderedDict

#Temporary
from utils.preprocessing import cleansing_data



LOGGER = get_logger(__name__)

dataset_column = []


MODELING_DEMO = OrderedDict(
    [
        ("---", startup),
        ("Classification", classification_modeling),
        ("Regression", regression_modeling),
        ("Clustering", clustering_modeling),
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
    target = st.selectbox("Choose Target column : ", dataset_df.columns.tolist())

    #Cleansing dataset
    any_dataframe = dataset_df.empty == False
    if any_dataframe:
        dataset_df = cleansing_data(dataset_df, target)
        st.subheader(f'Preprocessed Dataset')
        st.dataframe(dataset_df)

    if modeling_demo_name != "---" and not dataset_df.empty:
        demo_app(dataset_df, dataset_column)
    elif modeling_demo_name != "---" and dataset_df.empty:
        st.sidebar.error("Please upload data set first")
    else:
        demo_app()


if __name__ == "__main__":
    ml_demo_interface()
