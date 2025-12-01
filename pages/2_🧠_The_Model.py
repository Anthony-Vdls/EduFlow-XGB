
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score

st.set_page_config(page_title="The Model", page_icon="🧠")


@st.cache_resource
def train_reduced_model():
    """Recreate the reduced XGBoost model and compute metrics + feature importances."""
    # --- load tuned params & selected features ---
    with open("./data/best_params.json") as f:
        best_params = json.load(f)
    with open("./data/top_features.json") as f:
        top_features = json.load(f)

    # --- load raw NSCG microdata ---
    df = pd.read_csv("data/epcg23.csv")

    # --------- 1) construct y (same as in notebooks) ----------
    y_variables = ['DGRDG', 'WRKG', 'SALARY', 'OCEDRLP', 'DGRYR', 'STRTYR', 'STRTMN', 'HDMN']

    # months between graduation and job start
    months = (df['STRTYR'] - df['DGRYR']) * 12 + (df['STRTMN'] - df['HDMN'])

    df['y'] = (
        (df['DGRDG'] == 1) &                       # highest degree is bachelor's
        (df['WRKG'] == 'Y') &                      # working for pay
        (df['SALARY'] >= 1) & (df['SALARY'] < 9_999_998) &  # positive, non-topcoded salary
        (pd.to_numeric(df['OCEDRLP'], errors='coerce').isin([1, 2])) &  # job related to degree
        (months.between(0, 12, inclusive='both'))  # started job within ~1 year of graduation
    ).astype(np.float32)

    # --------- 2) restrict to recent bachelor’s grads ----------
    keep = (df['DGRDG'] == 1) & (df['DGRYR'] >= 2021)
    df = df.loc[keep].copy()

    # drop variables directly used in label definition
    df = df.drop(columns=[c for c in y_variables if c in df.columns])

    # --------- 3) drop leakage variables (post-outcome / admin) ----------
    leak_vars = [
        # 1) Direct label vars (plus near equivalents)
        "DGRDG", "DGRYR", "HDMN", "STRTYR", "STRTMN", "WRKG", "SALARY", "OCEDRLP",
        "NRCHG", "NRCON", "NRFAM", "NRLOC", "NROCNA", "NROT", "NRPAY", "NRREA", "NRSEC",

        # 2) Job status / employment history
        "HRSWK", "WKSLYR", "WKSWK", "WKSYR", "LFSTAT", "LOOKWK",
        "LWMN", "LWYR", "LWNVR",
        "NWFAM", "NWILL", "NWLAY", "NWNOND", "NWOCNA", "NWOT", "NWRET", "NWRTYR", "NWSTU",
        "PJFAM", "PJHAJ", "PJHRS", "PJNOND", "PJOCNA", "PJOT", "PJRET", "PJRETYR", "PJSTU",
        "FTPRET", "FTPRTYR", "WRKGP",

        # 3) Job satisfaction & benefits
        "JOBSATIS", "SATADV", "SATBEN", "SATCHAL", "SATIND", "SATLOC",
        "SATRESP", "SATSAL", "SATSEC", "SATSOC",
        "JOBINS", "JOBPENS", "JOBPROFT", "JOBVAC",

        # 4) Work activities
        "ACTCAP", "ACTDED", "ACTMGT", "ACTRD", "ACTRD2", "ACTRDT", "ACTRES", "ACTTCH",
        "WAACC", "WAAPRSH", "WABRSH", "WACOM", "WADEV", "WADSN", "WAEMRL", "WAMGMT", "WAOT",
        "WAPRI", "WAPROD", "WAPRRD", "WAPRSM", "WAPRSM2", "WAPRSM3", "WAQM", "WASALE",
        "WASCSM", "WASCSM2", "WASCSM3", "WASVC", "WATEA", "WASEC",

        # 5) Employer & occupation (post-outcome job details)
        "N2OCPRBG", "N2OCPRMG", "N3OCPR", "N3OCPRNG", "N3OCPRX",
        "N2OCBLST", "N2OCMLST", "N3OCLST", "N3OCLSTX", "N3OCNLST",
        "INDCODE", "EMED", "EMTP", "EMSECDT", "EMSECSM", "EMSIZE",
        "EMST_TOGA", "EMUS", "EMRG", "NEDTP", "NEWBUS", "PBPR21C",
        "CARN21C", "MGRNAT", "MGROTH", "MGRSOC", "SUPDIR", "SUPIND",
        "SUPWK", "TELEC", "TELEFR", "PJWTFT", "PRMBR", "PROMTGI",

        # 6) Training & coursework during/after job
        "WKTRNI", "WTRCHOC", "WTREASN", "WTREM", "WTRLIC", "WTROPPS", "WTROT",
        "WTRSKL", "WTRPERS",
        "ACADV", "ACCAR", "ACCCEP", "ACCHG", "ACDRG", "ACEM",
        "ACFPT", "ACGRD", "ACINT", "ACLIC", "ACOT", "ACSIN",
        "ACSKL", "NACEDMG", "NACEDNG",

        # 7) Survey design / admin
        "OBSNUM", "SURID", "SRVMODE", "WTSURVY", "COHORT", "REFYR",
        "BIRYR", "TCDGCMP",
    ]
    df = df.drop(columns=[c for c in leak_vars if c in df.columns])

    # --------- 4) convert objects → numeric (keep NaNs) ----------
    yn_map = {'Y': 1, 'N': 0, 'y': 1, 'n': 0}
    cols_to_drop = []

    for col in df.columns:
        if df[col].dtype == 'object':
            s = df[col].replace(yn_map)
            converted = pd.to_numeric(s, errors='coerce')
            if converted.notna().sum() == 0:
                cols_to_drop.append(col)
            else:
                df[col] = converted

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # cast numeric to float32
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].astype('float32')

    # --------- 5) build reduced feature matrix & split ----------
    X_small = df[top_features].copy()
    y = df['y'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y,
        test_size=0.2,
        random_state=67,
        stratify=y
    )

    # --------- 6) train reduced XGBoost classifier ----------
    model = XGBClassifier(
        objective="binary:logistic",
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        random_state=67,
        **best_params
    )

    model.fit(X_train, y_train)

    # --------- 7) evaluation ----------
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "log_loss": float(log_loss(y_test, probs)),
        "auc": float(roc_auc_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
    }

    feat_imp = pd.Series(model.feature_importances_, index=top_features).sort_values(ascending=False)

    return model, top_features, metrics, feat_imp


####################################################################

st.title("🧠 The Model")

st.markdown(
    """
This page explains how the model behind **“See Your Chances”** was built.

The goal is to estimate, for recent bachelor’s graduates, the probability that they’ll
be working **in a job related to their most recent degree** within about a year of graduating.
"""
)

# catched information from page 3
with st.spinner("Training the reduced model (cached)…"):
    model, top_features, metrics, feat_imp = train_reduced_model()

# 1
st.header("1. What the model predicts")

st.markdown(
    """
We built a **binary classifier** (XGBoost) that predicts a probability between 0 and 1:

- **1 =** working **for pay**, in a job **related to your field of study**, and that job **started within ~1 year of earning your bachelor’s degree**.
- **0 =** everyone else (who are just recent bachelors).

The data comes from the **National Survey of College Graduates (NSCG) 2023** microdata.
We restricted the dataset to people whose **highest degree is a bachelor’s** and who
earned it in **2021 or later**.
"""
)

# 2 
st.header("2. Defining the target (y)")

st.markdown(
    """
A respondent is labeled **1**(in-feild) if *all* of the following are true:

- Their **highest degree is a bachelor’s** (`DGRDG = 1`).
- They were **working for pay** during the survey(`WRKG = 'Y'`).
- Their **salary** is at least \$1 and not a top-coded special value.
- They reported their **principal job is related to their highest degree**  
  (`OCEDRLP` in {1 = "closely related", 2 = "somewhat related"}).
- The **job start date is within 0–12 months** after the degree date  
  (computed months from `DGRYR/HDMN` to `STRTYR/STRTMN`).

Everyone else is labeled **0 (not in-field) otherwise**.
"""
)

# 3 
st.header("3. Avoiding data leakage")

st.markdown(
    """
To keep the model honest, **variables were removed that revealed the outcome directly** or
only exist *because* of the outcome (having a job). 

Examples:

- **Label composition**: working status, job start year/month, salary, and
  the "job related to degree" flag itself.
- **“Reasons working outside field” items** (`NR*`): only asked when the job is not related,
  so they basically say *“this is a 0”*.
- **Job satisfaction and benefits**: things like satisfaction with salary, security, or
  social contribution are *results* of the job you got.
- **Detailed occupation / industry / employer info**: we wouldn’t know your exact job/industry
  at prediction time, so we don’t let the model use it.

We also dropped survey IDs and other admin variables, which are about how the survey was run, not about the graduate taking the survey.
"""
)

# 4   
st.header("4. From 500+ features to a reduced model")

st.markdown(
    """
The raw dataset has **hundreds of variables**.  
To make this usable as an interactive predictor this had to be **drastically** reduced. 

1. Train an initial XGBoost model on a cleaned, de-leaked version of all features.
2. Use the model’s **feature importances** to see which variables actually matter.
3. Select a reduced set of **top features** (24 that user inputs) that carry most of the signal.
4. Retrain a new XGBoost classifier using **only those reduced features**.

This reduced model:

- is **simpler to interpret**,  
- is easier to turn into an interactive app with just a handful of questions, and  
- and it actually **performed better** than the bigger model!
  (see repository notebooks for more details).
"""
)

st.subheader("Top features used in the final model")

st.code(", ".join(top_features), language="text")

# 5 
st.header("5. How well does it perform?")

st.markdown("We evaluated the reduced model on a held-out test set (that was 20% of the data):")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Log Loss ↓", f"{metrics['log_loss']:.3f}")
with c2:
    st.metric("AUC (ROC) ↑", f"{metrics['auc']:.3f}")
with c3:
    st.metric("F1 Score ↑", f"{metrics['f1']:.3f}")

st.markdown(
    """
- **Log loss**: What was **trying** to be minimized, measures how well-calibrated the probabilities are
  (lower is better).
- **AUC**: measures how well the model ranks in-field vs not-in-field cases  
  (0.5 = random, 1.0 = perfect).
- **F1**: balances precision and recall for the 1-class.
"""
)

# 6 
st.header("6. Which features matter most?")


st.markdown(
    """
Below are the **top 15 features** by importance in the reduced XGBoost model.
Higher bars mean the model relied more on that feature when making splits.
"""
)

# taken from catch 
top_k = min(15, len(feat_imp))
imp_df = feat_imp.head(top_k).reset_index()
imp_df.columns = ["feature", "importance"]

# Map short codes -> human-friendly labels for the x-axis
feature_label_map = {
    "GOVSUP":   "Federal support for your work",
    "EARN":     "Earned income last year ($)",
    "HDACY3":   "Year highest degree earned (3-yr group)",
    "ND2MENG":  "Field of 2nd highest degree (minor group)",
    "CH6":      "Children under age 6 (count)",
    "HDGRD":    "Got highest degree to prepare for grad school",
    "MRGRD":    "Got most recent degree to prepare for grad school",
    "CCST_TOGA":"Location of community college (assoc. degree)",
    "CH1218":   "Children age 12–18 (count)",
    "UGFEM":    "Employer helped pay for undergrad",
    "CH25":     "Children age 2–5 (count)",
    "HSYR":     "Year finished high school",
    "AGE":      "Age",
    "NBAMEBG":  "Broad field of bachelor’s degree",
    "CHU2":     "Children under age 2 (count)",
    "CLICNOW":  "License/cert is for intended job",
    "NMRMENG":  "Field of most recent degree (minor group)",
    "FACSEC":   "Importance of job security",
    "FSHHS":    "Federal support from HHS",
    "UGFPLN":   "Used loans from parents/relatives for undergrad",
    "CLICEM":   "License/cert required by employer",
    "D2PBP21C": "2nd highest degree institution type",
    "N2ACED":   "Current degree field (major group)",
    "NDGMEMG":  "Field of highest degree (major group)",
    "CCCOLPR":  "Attended community college to prep for college",
    "CHU2IN":   "Any children under age 2",
    "N2ACEDX":  "Reported current degree field",
    "NATIVE":   "American Indian / Alaska Native",
    "CHUN12":   "Any children under age 12",
    "N2D2MEDX": "Reported field of 2nd highest degree",
}

imp_df["feature_label"] = imp_df["feature"].map(
    lambda f: feature_label_map.get(f, f)
)

# Build Plotly bar chart
fig = px.bar(
    imp_df,
    x="feature_label",
    y="importance",
)

fig.update_layout(
    xaxis_title="Feature",
    yaxis_title="Importance",
    xaxis_tickangle=-40,
    template="plotly_white",
    margin=dict(l=40, r=20, t=40, b=140),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
These features capture:

- **Educational trajectory** (fields of degree, reasons for study, community college use)
- **Demographics** (age, children in the household)
- **Financial context** (earnings, how undergrad was funded)

The **“See Your Chances”** page takes user inputs for these same features,
feeds them into this reduced model, and uses the resulting probability as your
estimated chance of being in an in-field job within about a year of graduating.
"""
)


st.header("7. Limitations of this model")

st.markdown(
    """
**1. Its a sample of people who choose to take the survey.** 

Like most survey data its subject to volunteer bias. The choice of doing this survey could be tied to thing that this survey is measuring, and this case it is what is most likely happening. Either a person doesnt fill out this survey because they are not satisified with there schooling, or they are to busy from all their success to do so. The first scenerio is more likely

---

**2. It captures correlations, not causes.**  

The model learns **associations** between features (age, field, earnings, family situation, etc.)
and being in-field.  
It cannot answer *“if I change X, will Y definitely improve?”*.  
There are always unobserved factors like  networking, timing, and luck are not modeled.

---

**3. This model will not age well unless updated.**   

The data reflect graduates and labor market conditions **around 2023**.  
Job markets, demand by field, and the impact of AI and remote work can change over time.  
Unless the model is retrained on newer data, its accuracy will degrade.  
Good news - with this pipeline that made this model, it wouldnt be to hard. 

"""
)

