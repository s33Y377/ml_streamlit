import io
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper

uploadedFile = st.file_uploader("Upload a file")
if uploadedFile:
    df = pd.read_csv(uploadedFile)
    st.write(df)
    column_list = df.columns.to_list()
    column_lable = st.selectbox("Select Label Column?", df.columns.to_list())
    # fetaure and label split
    label = df.pop(column_lable)
    features = df
    st.write(features)

    st.write(features.dtypes)

    st.subheader("Apply Preprocessing")
    categorical_columns = st.multiselect(
        "Select Categorical Columns", column_list, None  # Default Selected
    )
    normalisation_columns = st.multiselect(
        "Select Numerical Column to apply Normalization",
        column_list,
        None,  # Default Selected
    )

    # determine categorical and numerical features
    numerical_ix = features.select_dtypes(include=["int64", "float64"]).columns
    categorical_ix = features.select_dtypes(include=["object", "bool"]).columns
    if st.button("Apply"):
        # define the data preparation for the columns
        column_transformer = [
            ("cat", OrdinalEncoder(), categorical_ix),
            ("num", StandardScaler(), numerical_ix),
        ]
        col_transform = ColumnTransformer(transformers=column_transformer)
        df = col_transform.fit_transform(df)
        # fetaure[categorical_columns] = fetaure[categorical_columns].apply(LabelEncoder().fit_transform)
        # fetaure[normalisation_columns] = fetaure[normalisation_columns].apply(StandardScaler().fit_transform)
        st.write(df)
    else:
        pass

    st.subheader("Apply Data Preprocesssing")
    permit_preprocess = st.checkbox("I agree.")
    st.subheader("Heat Map for Data Correlation")
    fig, ax = plt.subplots()
    sns.heatmap(features.corr(), ax=ax, annot=True)
    st.write(fig)
else:
    st.info("Please Upload a csv file with feature and label")
