import pandas as pd
from xgboost import XGBRegressor as xgbr
from xgboost import XGBClassifier as xgbc
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, f1_score, roc_auc_score, log_loss

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="The Model",
    page_icon="🧠",
    layout="wide"
)

st.title("Gradient Boosted Decision Tree")
st.markdown('---')
st.markdown(
    """
    """
)

st.markdown('---')

