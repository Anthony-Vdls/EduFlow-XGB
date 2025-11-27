import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="🖥️",
    layout="wide"
)

st.title("Education Exploration")
st.markdown("Source code for this website: [here](https://github.com/Anthony-Vdls/EduFlow-XGB)")
st.markdown('---')
st.markdown(
    """
    ## Navigate
    Thank you for visiting. Navigate this page by the left sidebar (>> on mobile).  
    There you will find:  
    - **💽About the Data** - Where it came from, what the aim is.
    - **🧠The Model** - The Machine Learning model, how it was made, and the features.
    - **🎲See Your Chances** - Make predictions with this model!
    """
)

st.markdown('---')
st.markdown(
    """
    ## Objective
     The objective of this machine learning model is to predict the probability of a recent batchelor graduate getting a job in their field within a year after graduation. This website's aim is to guide you through the begining to the end of the making of this model, starting with an introduction to the data, then the building of the model, and finally using the model to make predictions.

    """
)
