import json
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(
    page_icon="🎲",
    layout="wide"
)

# A WHOLE LOTTA codes needed for edu section
MAJOR_GROUPS = {
    1: "Computer & mathematical sciences",
    2: "Biological, agricultural & environmental life sciences",
    3: "Physical & related sciences",
    4: "Social & related sciences",
    5: "Engineering",
    6: "S&E-related fields",
    7: "Non-S&E fields",
    8: "Logical skip / not enrolled",
}

MINOR_GROUPS = {
    11: "Computer & information sciences",
    12: "Mathematics & statistics",
    21: "Agricultural & food sciences",
    22: "Biological sciences",
    23: "Environmental life sciences",
    31: "Chemistry (except biochemistry)",
    32: "Earth, atmospheric & ocean sciences",
    33: "Physics & astronomy",
    34: "Other physical sciences",
    41: "Economics",
    42: "Political & related sciences",
    43: "Psychology",
    44: "Sociology & anthropology",
    45: "Other social sciences",
    51: "Aerospace / aeronautical / astronautical engineering",
    52: "Chemical engineering",
    53: "Civil & architectural engineering",
    54: "Electrical & computer engineering",
    55: "Industrial engineering",
    56: "Mechanical engineering",
    57: "Other engineering",
    61: "Health",
    62: "Science & math teacher education",
    63: "Technology & technical fields",
    64: "Other S&E-related fields",
    71: "Management & administration fields",
    72: "Education (non–sci/math teacher ed.)",
    73: "Social service & related fields",
    74: "Sales & marketing fields",
    75: "Arts & humanities fields",
    76: "Other non-S&E fields",
    98: "Logical skip / no such degree",
}

FIELD_OF_STUDY_CODES = {
    0: "Other / not listed / no degree",
    116730: "Computer science",
    128420: "Mathematics, general",
    226320: "Biology, general",
    318730: "Chemistry (except biochemistry)",
    338780: "Physics (except biophysics)",
    419230: "Economics",
    429280: "Political science & government",
    438940: "Psychology, general",
    527250: "Chemical engineering",
    547280: "Electrical / electronics / comms engineering",
    567350: "Mechanical engineering",
    617860: "Medicine, dentistry, optometry, veterinary, etc.",
    716530: "Business administration & management",
    738620: "Philosophy / religion / theology",
    757600: "English language & literature",
    758200: "Liberal arts & sciences",
    768100: "Legal professions & studies",
    769950: "Other fields (not listed)",
}

CCCOLPR_OPTIONS = {
    0: "No",
    1: "Yes – mainly to prepare for a 4-year college",
}

D2PBP21C_OPTIONS = {
    0: "No second-highest degree / not sure",
    1: "Public institution",
    2: "Private institution",
}

CCST_TOGA_OPTIONS = {
    0: "Did not earn an associate’s degree",
    99: "United States (any state / territory)",
    100: "Outside U.S. (other country)",
}

def select_with_labels(label, mapping, default_code, key=None):
    """Helper method, returns the numeric code, shows 'code: label'."""
    codes = list(mapping.keys())
    try:
        default_idx = codes.index(default_code)
    except ValueError:
        default_idx = 0

    return st.selectbox(
        label,
        options=codes,
        index=default_idx,
        format_func=lambda c: f"{c}: {mapping[c]}",
        key=key,
    )

@st.cache_resource
def load_model():
    """
    This remakes everything in the deterministic_model notebook
    And returns an ordered list of the features used in it
    """
    # load params from notebook models
    with open("./data/best_params.json") as f:
        best_params = json.load(f)
    with open("./data/top_features.json") as f:
        top_features = json.load(f)


    # load data 
    df = pd.read_csv("data/epcg23.csv")

    required_cols = ['STRTYR', 'DGRYR', 'STRTMN', 'HDMN', 'DGRDG', 'WRKG', 'SALARY', 'OCEDRLP']

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(
            "The data file `data/epcg23.csv` on this server is missing required "
            f"columns: {missing}. This usually means the CSV in the deployed app "
            "is not the same as the one you used locally."
        )
        st.write("Here are the columns I *do* see in the deployed CSV:")
        st.write(list(df.columns))
        st.stop()

    # label y
    y_variables = ['DGRDG','WRKG','SALARY','OCEDRLP','DGRYR','STRTYR','STRTMN','HDMN']

    months = (df['STRTYR'] - df['DGRYR']) * 12 + (df['STRTMN'] - df['HDMN'])

    df['y'] = (
        (df['DGRDG'] == 1) &
        (df['WRKG'] == 'Y') &
        (df['SALARY'] >= 1) & (df['SALARY'] < 9_999_998) &
        (pd.to_numeric(df['OCEDRLP'], errors='coerce').isin([1, 2])) &
        (months.between(0, 12, inclusive='both'))
    ).astype(np.float32)

    # keep only recent bachelors
    keep = (df['DGRDG'] == 1) & (df['DGRYR'] >= 2021)
    df = df.loc[keep].copy()

    # drop variables used directly in y
    df = df.drop(columns=[c for c in y_variables if c in df.columns])

    # --------- drop leak variables ----------
    leak_vars = [
        # 1) Direct label vars 
        "DGRDG","DGRYR","HDMN","STRTYR","STRTMN","WRKG","SALARY","OCEDRLP",
        "NRCHG","NRCON","NRFAM","NRLOC","NROCNA","NROT","NRPAY","NRREA","NRSEC",

        # 2) Job status / employment
        "HRSWK","WKSLYR","WKSWK","WKSYR","LFSTAT","LOOKWK","LWMN","LWYR","LWNVR",
        "NWFAM","NWILL","NWLAY","NWNOND","NWOCNA","NWOT","NWRET","NWRTYR","NWSTU",
        "PJFAM","PJHAJ","PJHRS","PJNOND","PJOCNA","PJOT","PJRET","PJRETYR","PJSTU",
        "FTPRET","FTPRTYR","WRKGP","SURV_SE","EDTP",

        # 3) Job satisfaction & benefits
        "JOBSATIS","SATADV","SATBEN","SATCHAL","SATIND","SATLOC","SATRESP","SATSAL","SATSEC","SATSOC",
        "JOBINS","JOBPENS","JOBPROFT","JOBVAC",

        # 4) Work activities
        "ACTCAP","ACTDED","ACTMGT","ACTRD","ACTRD2","ACTRDT","ACTRES","ACTTCH",
        "WAACC","WAAPRSH","WABRSH","WACOM","WADEV","WADSN","WAEMRL","WAMGMT","WAOT",
        "WAPRI","WAPROD","WAPRRD","WAPRSM","WAPRSM2","WAPRSM3","WAQM","WASALE",
        "WASCSM","WASCSM2","WASCSM3","WASVC","WATEA","WASEC",

        # 5) Employer & occupation
        "N2OCPRBG","N2OCPRMG","N3OCPR","N3OCPRNG","N3OCPRX",
        "N2OCBLST","N2OCMLST","N3OCLST","N3OCLSTX","N3OCNLST",
        "INDCODE","EMED","EMTP","EMSECDT","EMSECSM","EMSIZE","EMST_TOGA","EMUS",
        "EMRG","NEDTP","NEWBUS","PBPR21C","CARN21C","MGRNAT","MGROTH","MGRSOC",
        "SUPDIR","SUPIND","SUPWK","TELEC","TELEFR","PJWTFT","PRMBR","PROMTGI",

        # 6) Training & courses after degree
        "WKTRNI","WTRCHOC","WTREASN","WTREM","WTRLIC","WTROPPS","WTROT","WTRSKL","WTRPERS",
        "ACADV","ACCAR","ACCCEP","ACCHG","ACDRG","ACEM","ACFPT","ACGRD","ACINT",
        "ACLIC","ACOT","ACSIN","ACSKL","NACEDMG","NACEDNG",

        # 7) Survey design / admin
        "OBSNUM","SURID","SRVMODE","WTSURVY","COHORT","REFYR","BIRYR","TCDGCMP",
    ]
    df = df.drop(columns=[c for c in leak_vars if c in df.columns])

    # convert to float32 dtypes
    yn_map = {'Y': 1, 'N': 0, 'y': 1, 'n': 0}
    cols_to_drop = []

    for col in df.columns:
        if df[col].dtype == 'object':
            s = df[col].replace(yn_map)
            converted = pd.to_numeric(s, errors='coerce')
            # drop feature if all NaNs
            if converted.notna().sum() == 0:
                cols_to_drop.append(col)
            else:
                df[col] = converted

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].astype('float32')

    # determinisic reduced model
    X_small = df[top_features].copy()
    y = df['y'].astype(int)

    model = XGBClassifier(
        objective="binary:logistic",
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        random_state=67,
        **best_params
    )
    model.fit(X_small, y)

    # we’ll also return column order for building new rows
    return model, top_features

def build_input_row(top_features):
    """
    Collects user inputs from the Streamlit widgets and returns
    a 1-row DataFrame with columns = top_features.
    """
    # init everything to 0.0 
    data = {feat: 0.0 for feat in top_features}

    st.subheader("Tell us a bit about you")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Your age", min_value=18, max_value=80, value=22)
        hsyear = st.number_input("Year you finished high school", min_value=1980, max_value=2030, value=2020)
        grad_year = st.number_input("Year you will complete / completed this bachelor’s", min_value=2019, max_value=2030, value=2024)

    with col2:
        native = st.checkbox("I identify as American Indian or Alaska Native")
        nbamebg = st.selectbox(
            "Broad field of your bachelor’s degree",
            # hard to make it not tuple
            options=[
                (1, "Science & Engineering"),
                (2, "S&E-related field"),
                (3, "Non-S&E field")
            ],
            index=0
        )
        prep_grad = st.checkbox("This degree was partly to prepare for graduate school / further education")
        

    # alot of kids questions 
    st.subheader("Number of kids you have in their age buckets")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ch_u2 = st.number_input("Children under 2", min_value=0, max_value=10, value=0)
    with c2:
        ch_2_5 = st.number_input("Children 2–5", min_value=0, max_value=10, value=0)
    with c3:
        ch_6_11 = st.number_input("Children 6–11", min_value=0, max_value=10, value=0)
    with c4:
        ch_12_18 = st.number_input("Children 12–18", min_value=0, max_value=10, value=0)

    # derived kids features
    ch6 = ch_u2 + ch_2_5 # under 6
    chu2in = 1 if ch_u2 > 0 else 0 # any under 2
    chun12 = 1 if (ch_u2 + ch_2_5 + ch_6_11) > 0 else 0

    # money stuff
    st.subheader("Money, work support & preferences")
    c1, c2, c3 = st.columns(3)
    with c1:
        earn = st.number_input("Total earned income last year (USD, before tax)", min_value=0, step=1000, value=0)
    with c2:
        ugfem = st.checkbox("My employer helped pay for my undergraduate studies")
        ugfpln = st.checkbox("I used loans from parents/relatives for undergrad")
    with c3:
        govsup_choice = st.radio(
            "Any of your work supported by the U.S. federal government?",
            ["No", "Yes", "Not sure"],
            index=0
        )
        govsup = {"No": 0, "Yes": 1, "Not sure / N/A": 0}.get(govsup_choice, 0)
        fshhs = st.checkbox("Any of that support came from HHS (Health & Human Services)")

    facsec_label = st.selectbox(
        "How important is job security to you?",
        options=[
            (1, "1 – Not important"),
            (2, "2 – Somewhat important"),
            (3, "3 – Important"),
            (4, "4 – Very important"),
        ],
        index=3
    )
    facsec = facsec_label[0]

    # cert
    st.subheader("Licenses or certifications (for your intended job)")
    has_cert = st.radio(
        "Do you have (or plan to have) at least one professional **certification** or **license** for your intended job? (Not bachelors degree)",
        ["No", "Yes"],
        index=0
    )
    if has_cert == "Yes":
        clicnow = st.checkbox("This certification/license is specifically for this job")
        clicem = st.checkbox("This certification/license is expected by the employer")
    else:
        clicnow = False
        clicem = False

    # education details
    st.subheader("Extra education details ")
    st.markdown(
        "These are code mappings used in the survey"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        ndgmemg = select_with_labels(
            "Field of *highest* degree (major group)",
            MAJOR_GROUPS,
            default_code=1,
            key="ndgmemg",
        )

        nmrmen = select_with_labels(
            "Field of *most recent* degree (minor group)",
            MINOR_GROUPS,
            default_code=11,
            key="nmrmen",
        )

        nd2meng = select_with_labels(
            "Field of *2nd highest* degree (minor group; choose 98 if you don't have one)",
            MINOR_GROUPS,
            default_code=98,
            key="nd2meng",
        )

    with c2:
        n2aced = select_with_labels(
            "Current degree field",
            FIELD_OF_STUDY_CODES,
            default_code=0,
            key="n2aced",
        )

        n2acedx = select_with_labels(
            "Reported current degree field (same codes)",
            FIELD_OF_STUDY_CODES,
            default_code=0,
            key="n2acedx",
        )

        n2d2medx = select_with_labels(
            "Reported field for 2nd highest degree",
            FIELD_OF_STUDY_CODES,
            default_code=0,
            key="n2d2medx",
        )

    with c3:
        cccolpr = select_with_labels(
            "Did you attend community college mainly to prepare for a 4-year college?",
            CCCOLPR_OPTIONS,
            default_code=0,
            key="cccolpr",
        )

        ccst_toga = select_with_labels(
            "Where was the community college that awarded your associate’s degree?",
            CCST_TOGA_OPTIONS,
            default_code=99,
            key="ccst_toga",
        )

        d2pbp21c = select_with_labels(
            "2nd highest degree institution type",
            D2PBP21C_OPTIONS,
            default_code=0,
            key="d2pbp21c",
        )


    # direct mappings
    data.update({
        "AGE": float(age),
        "HSYR": float(hsyear),
        "HDACY3": float(2019 if grad_year <= 2021 else 2022),
        "NATIVE": float(int(native)),
        "NBAMEBG": float(nbamebg[0]),
        "HDGRD": float(int(prep_grad)),
        "MRGRD": float(int(prep_grad)),
        "CHU2": float(ch_u2),
        "CH25": float(ch_2_5),
        "CH1218": float(ch_12_18),
        "CH6": float(ch6),
        "CHU2IN": float(chu2in),
        "CHUN12": float(chun12),
        "EARN": float(earn),
        "UGFEM": float(int(ugfem)),
        "UGFPLN": float(int(ugfpln)),
        "GOVSUP": float(govsup),
        "FSHHS": float(int(fshhs)),
        "FACSEC": float(facsec),
        "CLICNOW": float(int(clicnow)),
        "CLICEM": float(int(clicem)),
        "NDGMEMG": float(ndgmemg),
        "NMRMENG": float(nmrmen),
        "ND2MENG": float(nd2meng),
        "N2ACED": float(n2aced),
        "N2ACEDX": float(n2acedx),
        "N2D2MEDX": float(n2d2medx),
        "CCCOLPR": float(cccolpr),
        "CCST_TOGA": float(ccst_toga),
        "D2PBP21C": float(d2pbp21c),
    })

    # make sure all required columns exist
    row = pd.DataFrame([data])
    row = row[[c for c in top_features]]  # enforce order

    # types
    row = row.astype("float32")

    return row


# page layout
def main():
    st.title("Make Predictions with the Model")
    st.markdown(
        "Use the form below to enter your information. "
        "The model will estimate the probability that you’ll be working "
        "in a job related to your most recent bachelor’s degree within about a year of graduating. "
    )
    st.caption('Keep in mind that these are just predictions from a machine learning model, read about in **The Model** page before using.')

    model, top_features = load_model()

    with st.form("prediction_form"):
        input_row = build_input_row(top_features)
        submitted = st.form_submit_button("See My Chances 🎲")

    if submitted:
        prob = float(model.predict_proba(input_row)[0, 1])
        st.subheader("Your estimated chance of working **in-field**")
        st.metric(
            label="Predicted probability",
            value=f"{prob*100:0.1f}%"
        )

        st.progress(prob)

        
        if prob < 50:
            st.markdown(
                """
                This is **NOT** a guarantee or a personal judgment. Its a probability from a machine learning model based on patterns from the National Survey of College Graduates with similar situations. There is nothing that you cannot do with enough hardwork.
                """
            )
        else:
            st.markdown(
                """
                This is **NOT** a guarantee or a personal judgment. Its a probability from a machine learning model based on patterns from the National Survey of College Graduates with similar situations. 
                """
            )




if __name__ == "__main__":
    main()
