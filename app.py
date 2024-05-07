import streamlit as st
import pandas as pd
import joblib 
from GradientBoostingModel.train import PreprocessingData

trained_model = joblib.load('best_model_Gradient Boosting.pkl')

#defining streamlit app
def main():
    st.title('Customer Chrun Prediction')

    #input fields:
    Customer_Age = st.slider('Customer Age', min_value=18, max_value=140)
    gender_options = ['M','F']
    Gender = st.radio('Gender M = Male or F = Female', options=gender_options)
    Dependent_count = st.number_input('Dependent Count',min_value=0,max_value=5)
    Education_Level = st.selectbox('Education Level', ['Graduate', 'High School', 'Unknown','Educated',
                                                         'College', 'Post-Graduate', 'Doctorate'])
    Marital_Status = st.selectbox('Martial Status', ['Married', 'Single', 'Unknown', 'Divorced'])
    Income_Category = st.selectbox('Income Category',['Less than $40K', '$40K - $60K','$60K - $80K','$80K - $120K',
                                                      '$120K +','Unknown'])
    Card_Category = st.selectbox('Card Category', ['Blue', 'Gold','Silver','Platinum'])

    Months_on_book = st.slider('Months on Book', min_value=0, max_value=60)
    Total_Relationship_Count = st.slider('Total Relationship Count', min_value=0, max_value=10)
    Months_Inactive_12_mon = st.slider('Amount of months inactivity for last 12 months', min_value=0, max_value=12)
    Contacts_Count_12_mon = st.number_input('Number of contacts in the last 12 months',min_value=0,max_value=12)
    Credit_Limit = st.number_input('What is the credit limit?',min_value=0)
    Total_Revolving_Bal = st.number_input('total balans on credit card',min_value=0)
    Avg_Open_To_Buy = st.number_input('AVG open to buy value',min_value=0)
    Total_Amt_Chng_Q4_Q1 = st.number_input('Total_Amt_Chng_Q4_Q1', format="%0.3f")
    Total_Trans_Amt = st.number_input('Total_Trans_Amt',min_value=0)
    Total_Trans_Ct = st.number_input('Total_Trans_Ct',min_value=0)
    Total_Ct_Chng_Q4_Q1 = st.number_input('Total_Ct_Chng_Q4_Q1',min_value=0, max_value=2, format="%0.2f")
    Avg_Utilization_Ratio = st.number_input('Avg_Utilization_Ratio', min_value=0, max_value=1, format="%0.3f")
    

    # When 'Predict' button is clicked
    if st.button('Predict'):
        # Create a DataFrame with user input
        data = {
            'Customer_Age': [Customer_Age],
            'Gender': [Gender],
            'Dependent_count': [Dependent_count],
            'Education_Level': [Education_Level],
            'Marital_Status': [Marital_Status],
            'Income_Category': [Income_Category],
            'Card_Category': [Card_Category],
            'Months_on_book': [Months_on_book],
            'Total_Relationship_Count': [Total_Relationship_Count],
            'Months_Inactive_12_mon': [Months_Inactive_12_mon],
            'Contacts_Count_12_mon': [Contacts_Count_12_mon],
            'Credit_Limit': [Credit_Limit],
            'Total_Revolving_Bal': [Total_Revolving_Bal],
            'Avg_Open_To_Buy':[Avg_Open_To_Buy],
            'Total_Amt_Chng_Q4_Q1':[Total_Amt_Chng_Q4_Q1],
            'Total_Trans_Amt':[Total_Trans_Amt],
            'Total_Trans_Ct':[Total_Trans_Ct],
            'Total_Ct_Chng_Q4_Q1': [Total_Ct_Chng_Q4_Q1],
            'Avg_Utilization_Ratio':[Avg_Utilization_Ratio],
        }
        df = pd.DataFrame(data)

        #preprocessing data (as we did with the model)
        prep_data = PreprocessingData(df)
        prep_data.handle_categorical_values()

        #making a propability prediction
        prediction = trained_model.predict_proba(prep_data.data)[0][1]  # Probability of churn (attrited customer)

        #display prediction
        st.write(f"The probability that this client churns is: {prediction:.2%}")

if __name__ == "__main__":
    main()