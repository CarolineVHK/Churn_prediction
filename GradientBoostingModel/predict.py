import pandas as pd
import joblib
from train import PreprocessingData

print("one moment, the prediction is beiing calculated...")

#loading model:
gmb_model = joblib.load('best_model_Gradient Boosting.pkl')

#loading data using for prediction.
data_to_predict = pd.read_csv('/Users/caro/Documents/Projects/Churn_prediction/data/BankChurners.csv')

#preprocessing data:
prep_data = PreprocessingData(data_to_predict)
prep_data.drop_columns()
prep_data.handle_categorical_values()

#determine X to predict (drop column Attrition_Flag)
X_to_predict = prep_data.data.drop(columns=['Attrition_Flag'])

#make prediction:
prediction = gmb_model.predict(X_to_predict)

#print(prediction)
print('The predictions are: ', prediction)