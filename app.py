import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling as pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

# Load dataset if it exists
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar UI
with st.sidebar:
    st.title("Auto Machine Learning")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

# Upload section
if choice == "Upload":
    st.title("Upload Your Regression-Based CSV Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=False)  # Save the uploaded dataset
        st.dataframe(df)

# Profiling section
if choice == "Profiling":
    if df is not None:
        st.title("Exploratory Data Analysis")
        profile = pandas_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile)
    else:
        st.warning("Please upload a dataset first!")

# Modelling section
if choice == "Modelling":
    if df is not None:
        chosen_target = st.selectbox('Choose the Target Column', df.columns)

        # Handle missing values and non-numeric data before running PyCaret
        if st.button('Run Modelling'):
            # Step 1: Handle missing data by filling numeric columns with mean
            df = df.fillna(df.select_dtypes(include=['float64', 'int64']).mean())
            
            # Step 2: Convert non-numeric columns to categorical ones (one-hot encoding)
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                df[column] = df[column].astype(str)  # Ensure the column is of string type before encoding

            # Step 3: Apply one-hot encoding only to non-numeric columns
            df = pd.get_dummies(df, drop_first=True)  # Apply one-hot encoding
            
            # PyCaret setup
            setup(df, target=chosen_target)

            setup_df = pull()  # Retrieve the setup data
            st.dataframe(setup_df)

            # Model comparison
            best_model = compare_models()  # Get the best model
            compare_df = pull()  # Retrieve model comparison results
            st.dataframe(compare_df)

            # Save the best model
            save_model(best_model, 'best_model')

        else:
            st.warning("Please select a target column and click on 'Run Modelling'.")
    else:
        st.warning("Please upload a dataset first!")

# Download section
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model available for download. Please run the 'Modelling' section first.")
