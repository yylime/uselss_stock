import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from catboost import CatBoostRegressor




data = pd.read_csv("train.csv")
with open("selected_features_train.txt", "r") as f:
    selected_features = [item.replace('\n', '') for item in f.readlines()]
data = data[selected_features + ['target']]
print(data.info())

data = data.dropna(axis=0, subset = ["target"])

N = 5
fold = KFold(n_splits=N, shuffle=True, random_state=823)
features = [x for x in data.columns if x not in ['target']]
# features = selected_features never use this line

X = data[features].copy()
y = data['target']



if __name__ == "__main__":
    # data = pd.read_csv("train.csv")
    # print(data.info())
    pass