import streamlit as st
import pandas as pd 
import os 
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title('Auto ML')
    choice = st.radio('Navigation', ['Upload', 'Profiling', 'Modelling', 'Download'])
    st.info('This application allow you to build an automated ML pipeline.')

if os.path.exists('../data/dataset.csv'):
    df = pd.read_csv('../data/dataset.csv', index_col=None)

if choice == 'Upload':
    st.title('Upload Your Data for Modelling')
    file = st.file_uploader('Upload Your Dataset Here')

    if file:
        df = pd.read_csv(file)
        df.to_csv('../data/dataset.csv', index=None)
        st.dataframe(df)

if choice == 'Profiling':
    st.title('Exploratory Data Analysis')
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == 'Modelling':
    st.title('Training ML Model')
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    type = st.radio('Choose ML Category', ['Classification', 'Regression'])
    
    if type == 'Classification':
        from pycaret.classification import setup, compare_models, pull, save_model
    else:
        from pycaret.regression import setup, compare_models, pull, save_model

    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == 'Download':
    st.title('Download Best Model')
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")