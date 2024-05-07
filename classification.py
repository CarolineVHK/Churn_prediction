from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from GradientBoostingModel.train import PreprocessingData
from sklearn.model_selection import train_test_split
import pandas as pd

#defining preprocessing steps
preprocessing_steps= [('drop_columns', PreprocessingData().drop_columns()),
                      ('handle_categorical', PreprocessingData().handle_categorical_values()),
                      ('smote',SMOTE(random_state=42))]

#define model using: 
model_pipeline = XGBClassifier(random_state=42)

#create pipeline
pipeline = Pipeline(steps=preprocessing_steps + [('model', model_pipeline)])

#loading the data
data = pd.read_csv('./data/BankChurners.csv')
X = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

#split data to train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#fit pipeline
pipeline.fit(X_train, y_train)

#make prediction
prediction_pipline = pipeline.predict(X_test)



