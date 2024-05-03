import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import RandomForestRegressor

#load data and preprocessing
data = pd.read_csv('/data/BankChurners.csv')
X , y = data.iloc[:,:-1], data.iloc[:,-1]

#preprocessing the data
class Preprocessor():
    def __init__(self):
        pass

    def drop_columns():
        '''
        columns like Clientnumber is never used in Machin Learning, so we drop them to facilitate the data
        '''
        data.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
          "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
          'CLIENTNUM'], implace=True)
        
    def handle_unknown():
        '''
        How do we handle data with 'Unknown' as a value?
        Here I decided to ignore it for 
        '''
        pass
        
    def handle_categorical_values():
        '''
        Machine Learning models can't handle categorical values, so we need to encode then 
        (change into numerical data), I do thsi with the LabelEncoder from sklearn
        '''
        label_encoder = LabelEncoder()

        for col in data.columns:
            if data[col] == 'object':
                data[col] = label_encoder.fit_transform(data[col])
        
        


    
    