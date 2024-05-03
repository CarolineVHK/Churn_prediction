import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.model_selection import cross_val_score


#load data and preprocessing
data = pd.read_csv('/data/BankChurners.csv')
X = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

#preprocessing the data
class Preprocessor():
    def __init__(self, data):
        self.data = data

    def drop_columns(self, data):
        '''
        columns like Clientnumber is never used in Machin Learning, so we drop them to facilitate the data
        '''
        columns_to_drop = ["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
          "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
          'CLIENTNUM']
        data.drop(columns=columns_to_drop, inplace=True)
        return data
        
    def handle_unknown(self, data):
        '''
        How do we handle data with 'Unknown' as a value?
        Here I decided to ignore it for 
        '''
        pass
        
    def handle_categorical_values(self, data):
        '''
        Machine Learning models can't handle categorical values, so we need to encode then 
        (change into numerical data), I do thsi with the LabelEncoder from sklearn
        '''
        label_encoder = LabelEncoder()

        for col in data.columns:
            if data[col] == 'object':
                data[col] = label_encoder.fit_transform(data[col])
        return data
    
    def apply_smote():
        '''
        Since the data is out of balance (8500 Existing Customer over 1627 Attrited Customer) I decided to use 
        the SMOTE technique (see notebook for more info)
        '''
        smote = SMOTE()
        data = smote.fit(data)
        return data
        
# pipeline
preprocessor = Preprocessor()
pipeline = Pipeline([('preprocessor', preprocessor)])

#preprocessing data
X_preprocessed = pipeline.named_steps['preprocessor'].drop_columns(X)
X_preprocessed = pipeline.named_steps['preprocessor'].handle_unknown(X_preprocessed)
X_preprocessed = pipeline.named_steps['preprocessor'].handle_categorical_values(X_preprocessed)


#split data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


#i want to compaire different models to each other, so i create a function to evaluate the performance of each model
def evaluate_models(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_lable='blabla')
    recall = recall_score(y_test, y_pred, pos_lable='blabla')
    f1 = f1_score(y_test, y_pred, pos_lable='blabla')
    return accuracy, precision, recall, f1

#models I want to test
rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42)

#evaluation models
models = {'Random Forest':rf_model, 'KNN Model': knn_model, 'Logistric Regression': lr_model}
results = {}
for name_model, model in models.items():
    accuracy, precision, recall, f1 = evaluate_models(model, X_train, X_test, y_train, y_test)
    results[name_model] = {'accuracy':accuracy, 'precision': precision, 'recall': recall, 'f1':f1}

for name, scores in results.items():
    print(f"Model: {name}")
    for type,score in scores.items():
        print(f"{type} : {score}")
    print()
    
    