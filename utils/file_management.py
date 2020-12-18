import streamlit as st
import os

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a dataset (*csv)', filenames)
    return os.path.join(folder_path, selected_filename)