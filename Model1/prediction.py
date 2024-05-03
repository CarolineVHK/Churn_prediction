import pickle
import pandas as pd

from train import Preprocessor

data = pd.read_csv('./data/BankChurners.csv')
data_copy = data.copy()

with open('./model1/preprocess.pkl','rb') as f:
    preprocess = pickle.load(f)

X = preprocess.transform(data)

with open('./model1/model1.pkl','rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X)
data_predicted = pd.DataFrame({'Predicted':y_pred})
data = pd.concat([data_copy, data_predicted], axis=1)

data.to_csv('predicton.csv', index=False)
