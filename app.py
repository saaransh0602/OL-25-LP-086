#Importing libraries
import streamlit as st
import pandas as pd
import joblib
import xgboost

# Load models
df = pd.read_csv("Models & Datasets/cleaned_survey.csv")
clf_model = joblib.load("Models & Datasets/classification_model.pkl")
reg_model = joblib.load("Models & Datasets/regression_model.pkl")

# App layout
st.set_page_config(page_title="Mental Health App", layout="wide")


# Footer
def footer():
    st.markdown("---")
    st.markdown("""
    <small>Built with ❤️ by Saaransh Saxena 😎| 
    [LinkedIn](https://www.linkedin.com/in/saaransh-saxena-298256330/) • 
    [GitHub](https://github.com/saaransh0602) • 
    [X](https://x.com/itsSaaransh)</small>
    """, unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🏁 Exploratory Data Analysis",
        "📈 Regression Task",
        "🧮 Classification Task",
        "📊 Persona Clustering"
    ]
)

# 🏠 Home
if menu == "🏠 Home":
    st.title("Mental Wellness Analysis and Support Strategy in Tech Industry")
    st.divider()
    st.header("Dataset Overview")
    st.markdown("""
    ### Dataset Source: [Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    ### Collected by OSMI (Open Sourcing Mental Illness)
    ### Features include:
    * Demographic details (age, gender, country)
    * Workplace environment (mental health benefits, leave policies)
    * Personal experiences (mental illness, family history)
    * Attitudes towards mental health
    """)

    st.header("Problem Statement")
    st.markdown("""
        As a Machine Learning Engineer at NeuronInsights Analytics, you've been contracted by a coalition of
        leading tech companies including CodeLab, QuantumEdge, and SynapseWorks. Alarmed by rising burnout,
        disengagement, and attrition linked to mental health, the consortium seeks data-driven strategies to
        proactively identify and support at-risk employees. Your role is to analyze survey data from over 1,500 tech
        professionals, covering workplace policies, personal mental health history, openness to seeking help, and
        perceived employer support.
                    
        ### Project Objectives:
        * **Exploratory Data Analysis**
        * **Supervised Learning**:
            * *Classification task*: Predict whether a person is likely to seek mental health treatment (treatment column: yes/no)
            * *Regression task*: Predict the respondent's age
        * **Unsupervised Learning**: Cluster tech workers into mental health personas
        * **Streamlit App Deployment**
    """)

    footer()

# 🏁 Data Visualisation
elif menu == "🏁 Exploratory Data Analysis":
    st.title("📊 Data Analysis, Observations & Inferences")
    st.divider()
    st.write("This dataset had many anomalies, null values, outliers, and imbalanced data in columns like `Gender`, `Age`, `Country`, etc. which needed to be cleaned" \
    "and standardised.")
    st.write("Total number of values: `1259`")
    st.write("Total number of features: `27`")
    st.write("Features having NaN values: \n")
    st.write("\t`state`: 451")
    st.write("\t`self_employed`: 18")
    st.write("\t`work_interference`: 264")
    st.write("\t`comments`: 1095")
    st.divider()
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    st.divider()
    removed_features = ['Timestamp', 'Country', 'state', 'self_employed', 'comments']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Features Used:")
        for col in df.columns:
            st.markdown(f"- {col}")

    with col2:
        st.markdown("### ❌ Features Removed:")
        for col in removed_features:
            st.markdown(f"- {col}")

    st.divider()
    st.header("Univariate Analysis")
    st.image("Images/univariate1.png", caption="Univariate Analysis (1)", use_container_width=True)    
    st.image("Images/univariate2.png",  caption="Univariate Analysis (2)", use_container_width=True)
    st.image("Images/univariate3.png",  caption="Univariate Analysis (3)", use_container_width=True)

    st.divider()
    st.header("Bivariate Analysis")
    st.image("Images/bivariate.png", caption="Bivariate Analysis", use_container_width=True)
    
    st.divider()
    st.header("Multivariate Analysis")
    st.image("Images/multivariate2.png", caption="Correlation Heatmap)", use_container_width=True)
    st.image("Images/multivariate1.png", caption="Treatment Rate (%) based on Perceived Stigma", use_container_width=True)

    footer()

    
# 📈 Regression
elif menu == "📈 Regression Task":

    st.markdown("""
    **Models & Results**  
    | Model               | MAE        | RMSE      | R² Score    |
    |---------------------|------------|-----------|-------------|
    | Linear Regression   | 5.890833   | 7.365887  |  -0.036312  |
    | Random Forest       | 56.401374  | 7.510085  | -0.247727   |
    | XGBoost             | 52.247160  | 7.228220  | -0.155826   |

    🏆 **Best Model**: Linear Regression (by R² score)
    """)
    st.success("🏆 Linear Regression gives the best R² for age prediction, though all models have low explanatory power.")
    
    st.divider()
    # st.markdown("### This is a sample predictor of the age of a person given their conditions ⬇️")

    # input_dict_reg = {}
    # display_names_reg =  {
    # "self_employed": "Are you self-employed?",
    # "Gender": "Enter your Gender",
    # "family_history": "Do you have a family history of mental illness?",
    # "treatment": "Have you sought treatment for a mental health condition?",
    # "work_interfere": "If you have a mental health condition, do you feel that it interferes with your work?",
    # "no_employees": "What is the size of your company by number of employees?",
    # "remote_work": "Do you work remotely (outside of an office) at least 50% of the time?",
    # "benefits": "Does your employer provide mental health benefits?",
    # "care_options": "Do you know the options for mental health care your employer provides?",
    # "wellness_program": "Has your employer ever discussed mental health as part of an employee wellness program?",
    # "seek_help": "Does your employer provide resources to learn more about mental health issues and how to seek help?",
    # "anonymity": "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",
    # "leave": "How easy is it for you to take medical leave for a mental health condition?",
    # "mental_health_consequence": "Do you think that discussing a mental health issue with your employer would have negative consequences?",
    # "coworkers": "Would you be willing to discuss a mental health issue with your coworkers?",
    # "supervisor": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    # "obs_consequence": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
    # }

    # for col in df.columns:
    #     if col == "Age" or col == "age_group":  
    #         continue

    #     options = df[col].dropna().unique().tolist()
    #     label = display_names_reg.get(col, col) 
    #     input_dict_reg[col] = st.selectbox(label, options)

    # # Convert inputs into DataFrame
    # input_df = pd.DataFrame([input_dict_reg])

    # # Predict Age
    # if st.button("Predict Age"):
    #     # transformed = reg_pre.transform(input_df)
    #     predicted_age = reg_model.predict(df)

    #     st.success(f"🎯 Predicted Age: **{int(round(predicted_age[0]))} years**")

    footer()

# 🧮 Classification
elif menu == "🧮 Classification Task":
    st.title("🧮 Will the person seek treatment?")
    st.divider()
    st.markdown("The task at hand is to estimate wether the employee would seek help or not, making it a binary classification task. Below are the models used, and their evaluation results")
    
    st.header("📌 Classification: Will a person seek treatment?")
    st.markdown("""
    **Models & Results**  
    | Model                | Accuracy  |
    |----------------------|-----------|
    | Logistic Regression  | 0.765182  |
    | Random Forest        | 0.761134  |
    | XGBoost              | 0.769231  |

    🏆 **Best Model**: XGBoost (Accuracy: 0.769231)
    """)
    st.success("🏆 XGBoost outperforms Logistic Regression and Random Forest for treatment prediction.")

    st.divider()

    st.image("Images/ROC_Curve.png", caption="ROC Curve for different Classification models", use_container_width=False)

    st.divider()
    # st.markdown("### This is a sample predictor whether a person with given conditions is likely to seek mental health support or not ⬇️")

    # input_dict_clf = {}
    # display_names_clf = {
    # "self_employed": "Are you self-employed?",
    # "Gender": "Enter your Gender",
    # "family_history": "Do you have a family history of mental illness?",
    # "work_interfere": "If you have a mental health condition, do you feel that it interferes with your work?",
    # "no_employees": "What is the size of your company by number of employees?",
    # "remote_work": "Do you work remotely (outside of an office) at least 50% of the time?",
    # "benefits": "Does your employer provide mental health benefits?",
    # "care_options": "Do you know the options for mental health care your employer provides?",
    # "wellness_program": "Has your employer ever discussed mental health as part of an employee wellness program?",
    # "seek_help": "Does your employer provide resources to learn more about mental health issues and how to seek help?",
    # "anonymity": "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",
    # "leave": "How easy is it for you to take medical leave for a mental health condition?",
    # "mental_health_consequence": "Do you think that discussing a mental health issue with your employer would have negative consequences?",
    # "coworkers": "Would you be willing to discuss a mental health issue with your coworkers?",
    # "supervisor": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    # "obs_consequence": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
    # }

    # for col in df.columns:
    #     if col == "age_group":  # Skip this column
    #         continue

    #     if col == "treatment": 
    #         continue

    #     if col == "Age":
    #         input_dict_clf[col] = st.number_input("Enter your age", min_value=19, max_value=100, step=1)
    #     else:
    #         options = df[col].dropna().unique().tolist()
    #         label = display_names_clf.get(col, col)
    #         input_dict_clf[col] = st.selectbox(label, options)

    # input_df = pd.DataFrame([input_dict_clf])

    # # Predict 
    # if st.button("Predict"):
    #     input_df = pd.DataFrame([input_dict_clf]) 
    #     prediction = clf_model.predict(input_df)[0]

    #     # Step 4: Output result
    #     if prediction == 1:
    #         st.success("✅ Predicted: Will likely seek treatment!")
    #     else:
    #         st.error("❌ Predicted: Will likely not seek treatment!")

    footer()

# 📊 Clustering
elif menu == "📊 Persona Clustering":
    st.title("📊 Clustering Analysis")
    st.divider()
    st.markdown("The objective of this task is to make clusters and group tech workers according to their mental health personas. Below are some of the techniques and algorithms applied for the same.")
    st.write("The columns `Age`, `Country`, `Gender`, `no_employees` were dropped due to their less contribution in Mental Health Persona of an employee. These features" \
    "somewhere get covered in the rest of the questionnaire filled by the respondents.")

    # Clustering techniques
    st.subheader("Techniques Used: ")
    st.write(" - Principal Component Analysis (PCA)\n - t-distributed Stochastic Neighbor Embedding (t-SNE)\n - Uniform Manifold Approximation and Projection (UMAP)")
    st.write("Here is the plot for all three techniques applied on this dataset.")
    st.image("Images/reduce.png", caption="From these clusters we can see that `UMAP` forms the best and most clear and seggregated clusters out of the three.", use_container_width=True)
    
    st.write("The most optimal number of clusters were found to be 7, ranked by silhouette score for each cluster number.")
    st.image("Images/k_values.png", use_container_width=True)
    st.divider()

    st.markdown("### Here is the comparison (silhouette score) for all the models applied for clustering: ")
    st.write(" - **K-Means Clustering:** 0.4836\n - **Agglomerative Clustering:** 0.4619\n - **DBSCAN:** 0.2192 (13 DBSCAN Clusters, 2 Noise Points)")

    st.markdown("### ✅ From these scores, we can easily say that `K-Means` is clearly our winner.")
    st.image("Images/algos.png", caption="Clusters formed by the models", use_container_width=True)

    st.divider()
    st.image("Images/comparison.png", caption="Clusters formed by the models", use_container_width=True)
    st.divider()

    st.markdown("### 🧠 These are the different Personas the respondents can be classified into ⬇️")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Cluster 1", "Cluster 2", "Cluster 3",
    "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7"
    ])

    with tab1:
        st.markdown("""
        ### Cluster 1 – Balanced Support Seekers  
        Progressive yet moderate across all dimensions. Likely to participate in programs but not actively lead initiatives.
        """)

    with tab2:
        st.markdown("""
        ### Cluster 2 – Cautious Engagers  
        Low openness and remote work participation, moderate treatment rates. Engage only when trust is established.
        """)

    with tab3:
        st.markdown("""
        ### Cluster 3 – Policy-Dependent Advocates  
        Highly responsive to strong workplace benefits and leave policies. Thrive in structured, supportive environments.
        """)

    with tab4:
        st.markdown("""
        ### Cluster 4 – High-Risk Silent Sufferers  
        Strong treatment and family history indicators but low support scores. May avoid disclosure due to perceived stigma.
        """)

    with tab5:
        st.markdown("""
        ### Cluster 5 – Under-Supported Reluctants  
        Experience consequences without adequate access to treatment or leave. Likely constrained by organizational gaps.
        """)

    with tab6:
        st.markdown("""
        ### Cluster 6 – Stigma-Affected Minimalists  
        Engage little with mental health resources despite some awareness of physical health consequences.
        """)
    
    with tab7:
        st.markdown("""
        ### Cluster 7 – Empowered Open Advocates  
        High support scores, confident use of leave, and strong openness. Natural champions for wellness advocacy.
        """)

    footer()