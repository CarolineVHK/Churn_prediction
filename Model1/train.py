import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#load data and preprocessing
data = pd.read_csv('/data/BankChurners.csv')
X , y = data.iloc[:,:-1], data.iloc[:,-1]

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
    def handle_categorical_values():
        

    
    
    