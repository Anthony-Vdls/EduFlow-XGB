import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="💽",
    layout="wide"
)

st.title("About The Data")
st.markdown('---')


 
col1, col2 = st.columns([1, 2], vertical_alignment="center")

with col1:
    st.markdown(
        """
        Data: [Source](https://ncses.nsf.gov/explore-data/microdata/national-survey-college-graduates)  
        **Rows x Columns** :94606 x 548  

        >The National Survey of College Graduates (NSCG) is a federal survey from the National Center for Science and Engineering Statistics (NCSES) that collects data on U.S. residents with at least a bachelor’s degree, including many in STEM fields. It covers demographics, educational history, employment, wages, and job satisfaction, etc. Each biennial survey includes both returning respondents from prior waves and newly sampled respondents, and the data is pubic use and is available for download.
        """
)
    pass
with col2:
    st.markdown(
            """
            ## Y label: 
            We must engineer a new feature that we will be predicting. **Probability of recent bachelor graduates getting a paid job within their field after graduation.**  This will be a combination of these features:  
            1) **WRKG** Working for pay or profit during reference week
            2) **OCEDRLP** Extent that principal job is related to highest degree
            3) **DGRDG** Highest degree type
            4) **STRTYR** Year principal job started
            5) **DGRYR** Year of award of highest degree

            To make this label on the data these had to be true:  
            ```
            WRKG = Y
                - "Is working"
            OCEDRLP ∈ (1,2)
                - "Job is at least somewhat related to highest degree"
            DGRDG = 1
                - "Highest degree is a bachelors"
            STRTYR >= DGRYR
                - "Job was started AFTER they graduated"
            ```
            """
        )

st.markdown('---')

st.markdown(
        """
        ## Other relevent features
        Looking though all 500+ of the features these stand out as possibly stronger predictors:

### **Demograhic Related**
* **AGE** Age
* **SEX_2023** Sex at birth
* **CTZN** Citizenship or visa status
* **CTZFOR** Visa type for non-US citizens
* **FNUSYR6** The year first came to U.S. for 6 months or longer
* **VETSTAT** Veteran status: served on active duty in the US Armed Forces, Reserves, or National Guard
### **Geogrophy Related**
* **RESPLOC** Respondent location
* **RESPLO3_TOGA** 3-Digit Respondent Location (state/country code)
* **RESPLCUS** Respondent location (U.S./Non-U.S.)
* **EMRG** Region code for employer
* **EMST_TOGA** State/country code for employer
### **Finacial Related**
* **UGLOANR** Amount borrowed to finance UNDERGRADUATE degree(s)
* **UGOWER** Amount still owed from financing of UNDERGRADUATE degree(s)
* **GRFLN** Financial support for graduate degree(s): Loans from school, banks, and government
* **SALARY** Salary
### Other relevant features  
* **LFSTAT** Labor force status
* **DGRDG** Highest degree type
* **HDMN** Month of award of highest degree
* **NDGMEMG** Field of study for highest degree (major group)
* **NDGMENG** Field of study for highest degree (minor group)
* **HDPBP21C** Public/private status of school awarding highest degree - 2021 Carnegie code
* **HDRGN** Location of school awarding highest degree (region code)
* **BAYR** Year of award of first bachelors degree
* **CLICWKR** Certificates and licenses: for work-related reasons
* **CLICNOW** Certification or licenses: for principal job
* **CLICISS** Certification or licenses: issuer
* **CLICCODE** Certification/license primary subject or field of study
* **CLICYR** Certification or licenses: year first issued
        """
    )
