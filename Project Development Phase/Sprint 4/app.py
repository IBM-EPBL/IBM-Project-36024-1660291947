# Core Packages
import pickle
import requests
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
matplotlib.use('Agg')

st.set_page_config(page_title='UAEP', page_icon='images/logo.jpg',
                   layout='wide', initial_sidebar_state='auto')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# EDA Packages

sns.set(rc={'figure.figsize': (20, 15)})

DATA_URL = ('dataset/Admission_Predict.csv')
st.markdown('# UNIVERSITY ADMIT ELIGIBILITY PREDICTOR')

st.markdown('### **About the Project:**')
st.info('The aim of the web page is to help students in shortlisting \
               universities with their profiles. The predicted output gives them a fair idea\
                    about their admission chances in a  university. This analysis should also\
                          help students who are currently preparing or will be preparing to get a better idea. ')

st.markdown('### **METRICS**')
col1, col2, col3= st.columns(3)
col1.metric ("GRE Score","out of 340")
col2.metric ("TOEFL Score","out of 120")
col3.metric ("University Rating ","out of 5")

col4,col5,col6=st.columns(3)
col4.metric ("Statement of Purpose/SOP","out of 5")
col5.metric ("Letter of Recommendation/LOR","out of 5")
col6.metric ("CGPA","out of 10")

col7,col8,col9=st.columns(3)
col7.metric ("Research Experience","either 0 or 1")
col8.metric ("Chance of Admittance", "from 0 to 100%")
col9.metric ("Low chances","High chances")
st.markdown('### **Top universities all over the world**')

img = Image.open('images/gad.jpg')
st.image(img, width=720, caption='Top universities')
img = Image.open('images/univ.jpg')
st.image(img, width=720, caption="Top Universities in the US")


img = Image.open('images/par.png')
st.image(img, width=720, caption="Influence of the Attributes based on the metrics")


def load_data(nrows):
    df = pd.read_csv(DATA_URL, nrows=nrows)
    def lowercase(x): return str(x).lower()
    df.set_index('Serial No.', inplace=True)
    df.rename(lowercase, axis='columns', inplace=True)
    return df


st.title('Lets explore the University Admission Predictor')
# Creating a text element and let the reader know the data is loading.

data_load_state = st.text('Loading university admissions dataset...')
# Loading 500 rows of data into the dataframe.
df = load_data(500)

# Notifying the reader that the data was successfully loaded.
data_load_state.text('University admissions dataset successfully loaded...')
# Explore Dataset
st.sidebar.header('Dashboard')
st.sidebar.subheader('Explore the data')
st.header('Explore the data')
st.markdown("Select the data box on the side panel.")

if st.sidebar.checkbox("View Data"):
    st.subheader('View data')
    st.write(df)

if st.sidebar.checkbox('Dataset '):
    st.subheader('Dataset :')
    st.write(df.head())

if st.sidebar.checkbox("View Columns"):
    st.subheader('Show Columns List')
    all_columns = df.columns.to_list()
    st.write(all_columns)

if st.sidebar.checkbox(' Description'):
    st.subheader('Data Descripition')
    st.write(df.describe())

if st.sidebar.checkbox('Null Values?'):
    st.subheader('Null values')
    st.write(df.isnull().sum())

st.header('Data Visualization')
st.markdown("Select the box on the side panel.")
st.sidebar.subheader('Data Visualization')

if st.sidebar.checkbox('Distribution Plot'):
    st.subheader('Distribution Plot')
    st.info("If error, please adjust column name on side panel.")
    column_dist_plot = st.sidebar.selectbox('Choose a column to plot density.', df.columns[:5])
    fig = sns.distplot(df[column_dist_plot])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

if st.sidebar.checkbox('Categorical Plot'):
    st.subheader('Categorical Plot')
    st.info("If error, please adjust column name on side panel.")
    column_cat_plot = st.sidebar.selectbox("Choose a column to plot count.", df.columns[:6])
    fig = sns.catplot(x=column_cat_plot,data=df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Showing the Prediction Model
st.header(' Prediction Model for the admission')
st.sidebar.subheader('Prediction Model')
st.markdown("Select the box on the side panel.")


if st.sidebar.checkbox('Prediction'):
    st.subheader('Prediction Model')
    #pickle_in = open('model.pkl', 'rb')
    #model = pickle.load(pickle_in)
    
    API_KEY = "Zpck9ZE4GFOjymW3UwyodjQfkBvPEndSnhxkySdd9aMD"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    @st.cache()
    # defining the function to predict the output
    def convert_toefl_to_ielts(val):
        if val > 69 and val < 94:
            score = 6.5
        if val > 93 and val < 102:
            score = 7.0
        if val > 101 and val < 110:
            score = 7.5
        if val > 109 and val < 115:
            score = 8.0
        if val > 114 and val < 118:
            score = 8.5
        if val > 117 and val < 121:
            score = 9.0
        return score

    def pred(gre, toefl, sop, lor, cgpa, resc, univ_rank):

        # Preprocessing user input
        # ielts = convert_toefl_to_ielts(toefl)

        if resc == 'Yes':
            resc = 1
        else:
            resc = 0

        #Predicting the output
        #prediction = model.predict([[gre, toefl, univ_rank, sop, lor, cgpa, resc]])

        payload_scoring = {"input_data": [{"field": [["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA", "Research"]], 
        "values": [[gre, toefl, univ_rank, sop, lor, cgpa, resc]]}]}
        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/2179e430-17c3-431e-9be2-16715fdda1a2/predictions?version=2022-11-17', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})

        prediction = response_scoring.json()['predictions'][0]['values'][0][0]

        st.info("Chance of Admittance for University Rank " + str(univ_rank) + " = " + str(prediction[0]*100) +" %")
                # str(prediction[0]*100))
        if prediction[0] >= 0.6667:
            st.success('Congratulations! You are eligible to apply for the university!')
            chance = Image.open('images/chance.png')
            st.image(chance, width=300, caption="High Chances !")
        else:
            st.caption('Better Luck Next Time :)')
            no_chance = Image.open('images/nochance.png')
            st.image(no_chance, width=300, caption="Low Chances :(")

         # Main function for the UI of the webpage
    def main():

           # Text boxes in which user can enter data required to make prediction
           gre = st.number_input('GRE Score (out of 340):', min_value=0, max_value=340, value=260, step=1)
           toefl = st.number_input('TOEFL Score (out of 120):', min_value=0, max_value=120, value = 80, step=1)
           univ_rank = st.slider("University Rank (1 to 5):", value=1,
                        min_value=1, max_value=5, step=1)
           sop = st.slider("SOP Score (out of 5):", value=0.0,
                        min_value=0.0, max_value=5.0, step=0.5)
           lor = st.slider("LOR Score (out to 5):", value=0.0,
                      min_value=0.0, max_value=5.0, step=0.5)
           cgpa = st.number_input('Enter CGPA (out of 10):', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
           resc = st.selectbox('Research Experience:', ("Yes", "No"))

            # when 'Predict' is clicked, make the prediction and store it
           if st.button("Predict"):
             result = pred(gre, toefl, sop, lor, cgpa, resc, univ_rank)
 
    if __name__ == '__main__':
        main()

st.sidebar.subheader('Author Credits')
st.sidebar.info("[Athira J](https://github.com/jathira)\
    \n [Bernosha S B](https://github.com/BernoshaSB)\
    \n [DarshaGayathri K](https://github.com/darshagayathri)\
    \n [Dharshini S](https://github.com/sdhars)")
st.sidebar.subheader('Data Source')
st.sidebar.info("From the Kaggle")
st.sidebar.subheader('Built with Streamlit')
st.sidebar.info("https://www.streamlit.io/")