# Auto Machine Learning (AutoML) Streamlit App

## Overview

The **Auto Machine Learning (AutoML) Streamlit App** is an interactive web application that simplifies the machine learning workflow. Built using **Streamlit**, **PyCaret**, and **Pandas Profiling**, this app allows users to upload their regression-based datasets, perform comprehensive exploratory data analysis (EDA), automatically preprocess the data, train multiple machine learning models, and download the best-performing model. 

This project is designed to streamline the process of building machine learning models, enabling both data scientists and non-technical users to easily explore and analyze their datasets, compare various machine learning models, and select the most optimal one for prediction tasks.

### **Live Demo**
You can try the live demo of the application at: [AutoML Streamlit App](https://automl.streamlit.app/)

## Features

- **Dataset Upload**: Easily upload CSV files containing regression-based datasets for automatic processing.
- **Exploratory Data Analysis (EDA)**: Automatically generate detailed reports using **Pandas Profiling** to understand your data's distribution, correlations, and potential outliers.
- **Model Training**: Automatically preprocess the dataset (handle missing values, encode categorical variables) and train multiple regression models using **PyCaret**.
- **Model Comparison**: Compare the performance of various models, and automatically select the best one based on evaluation metrics.
- **Model Download**: Download the best model as a serialized `.pkl` file for future use or deployment.

## Tech Stack

- **Streamlit**: An open-source app framework for building interactive web applications.
- **PyCaret**: An open-source machine learning library that automates the process of training and evaluating models.
- **Pandas Profiling**: A tool to generate an EDA report from a Pandas DataFrame.
- **Plotly**: A library for creating interactive visualizations.
- **Pandas**: A data manipulation library for handling and analyzing structured data.

## Installation and Setup

Follow the steps below to run the app locally:

### 1. Clone the Repository
```
git clone https://github.com/your-username/automl-app.git
cd automl-app
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the App
```
streamlit run app.py
```
