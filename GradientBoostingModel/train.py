#import linrary for preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

#import library for models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#import library saving model
import joblib

#load data and preprocessing
data = pd.read_csv('./data/BankChurners.csv')


#preprocessing the data
class PreprocessingData:
    def __init__(self, data):
        self.data = data

    def drop_columns(self):
        '''
        columns like Clientnumber is never used in Machin Learning, so we drop them to facilitate the data
        '''
        columns_to_drop = ["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
          "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
          'CLIENTNUM']
        self.data.drop(columns=columns_to_drop, inplace=True)
        return self.data
        
    def handle_categorical_values(self):
        '''
        Machine Learning models can't handle categorical values, so we need to encode then 
        (change into numerical data), I do thsi with the LabelEncoder from sklearn
        '''
        label_encoder = LabelEncoder()

        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = label_encoder.fit_transform(self.data[col])
        return self.data


prep_data = PreprocessingData(data)
prep_data.drop_columns()
prep_data.handle_categorical_values()


#split data
X = prep_data.data.drop(columns=['Attrition_Flag'])
y = prep_data.data['Attrition_Flag']

#apply smote to handle imbalanced data: using SMOTE
'''
Since the data is out of balance (8500 Existing Customer over 1627 Attrited Customer) I decided to use 
the SMOTE technique (see notebook for more info)
'''
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X,y)

X_train, X_test, y_train, y_test= train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

#define model
gmb_model = XGBClassifier(random_state=42)                          #Gradient Boosting ML model

#save model:
joblib.dump(gmb_model,'Gradient_Boosting_model.pkl')

