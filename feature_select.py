from collections import Counter
import pandas as pd
from Trainer import Trainer
from sklearn.model_selection import train_test_split
import os
from pprint import pprint

### use catboost to select features

def select_by_catboost():
    dir = "data\selected_data"
    path = os.listdir(dir)
    for index, p in enumerate(path):
        p = os.path.join(dir, p)
        data = pd.read_csv(p, index_col=0)
        features = [x for x in data.columns if x not in ['y']]

        train_x = data[features].copy()
        train_y = data['y']

        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=823)

        trainer = Trainer(train_x=X_train,
                        train_y=y_train,
                        valid_x=X_test,
                        valid_y=y_test)
        _, model = trainer.catboost_train_with_valid()
        importance = model.feature_importances_
        names = model.feature_names_

        pd.Series(data=importance, index=names).to_csv(
            os.path.join("config\catfeature_importance", f"{index}.csv"))
    return None

def cal():
    dir = "config\catfeature_importance"
    path = os.listdir(dir)
    hs = set()
    for index, p in enumerate(path):
        h = []
        p = os.path.join(dir, p)
        data = pd.read_csv(p, index_col=0, header=0)
        for i in range(len(data)):
            h.append((data.index[i], data.iloc[i].values[0]))
        h = sorted(h, key=lambda x:-x[1])
        for i in range(min(30, len(h))):
            hs.add(h[i][0] + '\n')

    with open("config\selected_features.txt", 'w') as f:
        f.writelines(list(hs))
    pprint(len(hs))

def select_by_percent():
    with open(r"config/all_samples_features.txt", 'r') as f:
        lines = f.readlines()
    counter = Counter(lines)
    res = []
    for f, cnt in counter.items():
        if cnt >= 9:
            res.append(f)
    # with open("selected_features.txt", 'w') as f:
    #     f.writelines(res)
    print(len(res))
    res = []
    for f, cnt in counter.items():
        if cnt >= 10:
            res.append(f)
    with open("config\selected_features.txt", 'w') as f:
        f.writelines(res)

    print(len(res))

if __name__ == "__main__":
    # select_by_percent()
    # select_by_catboost()
    cal()