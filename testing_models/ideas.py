# #specifiing categoritical valeus
# attrition_mapping = {'Existing Customer': 1, 'Attrited Customer': 0}
# data['Attrition_Flag'] = data['Attrition_Flag'].map(attrition_mapping)


# gender_mapping = {'M': 0, 'F': 1}
# data['Gender'] = data['Gender'].map(gender_mapping)


# education_mapping = {'High School': 5, 'Graduate': 2, 'Uneducated': 1, 'Unknown': 0, 
#                      'College': 4, 'Post-Graduate': 3, 'Doctorate': 6}
# data['Education_Level'] = data['Education_Level'].map(education_mapping)


# marital_mapping = {'Married': 2, 'Single': 1, 'Unknown': 0, 'Divorced': 3}
# data['Marital_Status'] = data['Marital_Status'].map(marital_mapping)


# income_mapping = {'$60K - $80K': 3, 'Less than $40K': 1, '$80K - $120K': 4, 
#                   '$40K - $60K': 2, '$120K +': 5, 'Unknown': 0}
# data['Income_Category'] = data['Income_Category'].map(income_mapping)


# card_mapping = {'Blue': 0, 'Gold': 2, 'Silver': 1, 'Platinum': 3}
# data['Card_Category'] = data['Card_Category'].map(card_mapping)




#bespreking met Jens

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.base import BaseEstimator

# # from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# # from sklearn.model_selection import cross_val_score


# #load data and preprocessing
# data = pd.read_csv('/Users/caro/Documents/Projects/Churn_prediction/data/BankChurners.csv')
# print(data.head())
# X = data.drop(columns=['Attrition_Flag'])
# y = data['Attrition_Flag']

# #preprocessing the data
# class Preprocessor(BaseEstimator):
#     def __init__(self):
#         #self.data = data
#         pass

#     def fit(self, X, y=None):
#         return self

#     def drop_columns(self, data):
#         '''
#         columns like Clientnumber is never used in Machin Learning, so we drop them to facilitate the data
#         '''
#         columns_to_drop = ["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
#           "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
#           'CLIENTNUM']
#         data.drop(columns=columns_to_drop, inplace=True)
#         return data
        
#     def handle_unknown(self, data):
#         '''
#         How do we handle data with 'Unknown' as a value?
#         Here I decided to ignore it for 
#         '''
#         pass
        
#     def handle_categorical_values(self, data):
#         '''
#         Machine Learning models can't handle categorical values, so we need to encode then 
#         (change into numerical data), I do thsi with the LabelEncoder from sklearn
#         '''
#         label_encoder = LabelEncoder()

#         for col in data.columns:
#             if data[col] == 'object':
#                 data[col] = label_encoder.fit_transform(data[col])
#         return data
    
#     def apply_smote(self, data):
#         '''
#         Since the data is out of balance (8500 Existing Customer over 1627 Attrited Customer) I decided to use 
#         the SMOTE technique (see notebook for more info)
#         '''
#         smote = SMOTE()
#         data = smote.fit(data)
#         return data
    
#     def transform(self, X, y=None):
#         X = self.apply_smote(X)

#         return X

        
# # pipeline

# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X,y)
# print(X_res.head())
# # pipeline = Pipeline([('Smote', SMOTE())])
# # pipeline.fit(X)
# # data_2 = pipeline.transform(X)
# # print(data_2.head())



# #split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #i want to compaire different models to each other, so i create a function to evaluate the performance of each model
# def evaluate_models(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, pos_lable='blabla')
#     recall = recall_score(y_test, y_pred, pos_lable='blabla')
#     f1 = f1_score(y_test, y_pred, pos_lable='blabla')
#     return accuracy, precision, recall, f1

# #models I want to test
# rf_model = RandomForestClassifier(random_state=42)
# knn_model = KNeighborsClassifier(random_state=42)
# lr_model = LogisticRegression(random_state=42)

# #evaluation models
# models = {'Random Forest':rf_model, 'KNN Model': knn_model, 'Logistric Regression': lr_model}
# results = {}
# for name_model, model in models.items():
#     accuracy, precision, recall, f1 = evaluate_models(model, X_train, X_test, y_train, y_test)
#     results[name_model] = {'accuracy':accuracy, 'precision': precision, 'recall': recall, 'f1':f1}

# for name, scores in results.items():
#     print(f"Model: {name}")
#     for type,score in scores.items():
#         print(f"{type} : {score}")
#     print()




#voor bespreking met Jens:


# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression

# # from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# # from sklearn.model_selection import cross_val_score


# #load data and preprocessing
# data = pd.read_csv('/Users/caro/Documents/Projects/Churn_prediction/data/BankChurners.csv')
# print(data.head())
# X = data.drop(columns=['Attrition_Flag'])
# y = data['Attrition_Flag']

# #preprocessing the data
# class Preprocessor():
#     def __init__(self, data):
#         self.data = data

#     def drop_columns(self, data):
#         '''
#         columns like Clientnumber is never used in Machin Learning, so we drop them to facilitate the data
#         '''
#         columns_to_drop = ["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
#           "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
#           'CLIENTNUM']
#         data.drop(columns=columns_to_drop, inplace=True)
#         return data
        
#     def handle_unknown(self, data):
#         '''
#         How do we handle data with 'Unknown' as a value?
#         Here I decided to ignore it for 
#         '''
#         pass
        
#     def handle_categorical_values(self, data):
#         '''
#         Machine Learning models can't handle categorical values, so we need to encode then 
#         (change into numerical data), I do thsi with the LabelEncoder from sklearn
#         '''
#         label_encoder = LabelEncoder()

#         for col in data.columns:
#             if data[col] == 'object':
#                 data[col] = label_encoder.fit_transform(data[col])
#         return data
    
#     def apply_smote():
#         '''
#         Since the data is out of balance (8500 Existing Customer over 1627 Attrited Customer) I decided to use 
#         the SMOTE technique (see notebook for more info)
#         '''
#         smote = SMOTE()
#         data = smote.fit(data)
#         return data
        
# # pipeline
# preprocessor = Preprocessor(data)
# pipeline = Pipeline([('preprocessor', preprocessor)])

# #preprocessing data
# X_preprocessed = pipeline.named_steps['preprocessor'].drop_columns(X)
# X_preprocessed = pipeline.named_steps['preprocessor'].handle_unknown(X_preprocessed)
# X_preprocessed = pipeline.named_steps['preprocessor'].handle_categorical_values(X_preprocessed)


# #split data
# X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


# #i want to compaire different models to each other, so i create a function to evaluate the performance of each model
# def evaluate_models(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, pos_lable='blabla')
#     recall = recall_score(y_test, y_pred, pos_lable='blabla')
#     f1 = f1_score(y_test, y_pred, pos_lable='blabla')
#     return accuracy, precision, recall, f1

# #models I want to test
# rf_model = RandomForestClassifier(random_state=42)
# knn_model = KNeighborsClassifier(random_state=42)
# lr_model = LogisticRegression(random_state=42)

# #evaluation models
# models = {'Random Forest':rf_model, 'KNN Model': knn_model, 'Logistric Regression': lr_model}
# results = {}
# for name_model, model in models.items():
#     accuracy, precision, recall, f1 = evaluate_models(model, X_train, X_test, y_train, y_test)
#     results[name_model] = {'accuracy':accuracy, 'precision': precision, 'recall': recall, 'f1':f1}

# for name, scores in results.items():
#     print(f"Model: {name}")
#     for type,score in scores.items():
#         print(f"{type} : {score}")
#     print()

#_--> als ik smote wil gebruiken moet deze appart van de pipline gebeuren. daar het zijn eigen pipeline heeft.
#dus eerst de voorstapjes doen en dan smote en pas dan Pipeline.


    