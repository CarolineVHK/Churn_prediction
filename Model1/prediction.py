import pickle
import pandas as pd

from train import Preprocessor

data = pd.read_csv('./data/BankChurners.csv')

with open ('./model/prprocess.pkl','rb')