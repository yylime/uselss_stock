import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from catboost import CatBoostRegressor
import os

class Trainer:
    def __init__(self, train_x, train_y, valid_x, valid_y) -> None:
        self.x = train_x
        self.y = train_y
        self.vx = valid_x
        self.vy = valid_y

    def catboost_train_with_valid(self, model_path="trained_model"):

        oof = np.zeros((len(self.x), ))
        model = CatBoostRegressor(iterations=4000,
                                    depth=8,
                                    learning_rate=0.05,
                                    task_type="GPU",
                                    bagging_temperature=0.2,
                                    early_stopping_rounds=150,
                                    random_state=823)

        train_x, train_y = self.x, self.y
        valid_x, valid_y = self.vx, self.vy

        model.fit(train_x,
                    train_y,
                    eval_set=(valid_x, valid_y),
                    use_best_model=True)
        valid_predict = model.predict(valid_x)
        model.save_model(os.path.join(model_path, model_path))

        mean_squared_error = metrics.mean_squared_error(valid_y, valid_predict)
        print("均方误差为", mean_squared_error)
        return mean_squared_error, model


    def catboost_train(self, cv=5, model_path="trained_model"):
        fold = KFold(n_splits=cv, shuffle=True)
        oof = np.zeros((len(self.x), ))
        for index, (train_idx, valid_idx) in enumerate(fold.split(self.x, self.y)):
            # model
            model = CatBoostRegressor(
                loss_function = "MAE",
                iterations=4000,
                depth=10,
                learning_rate=0.05,
                task_type="GPU",
                # bagging_temperature=0.8,
                early_stopping_rounds=200,
                random_state=823
            )
            
            train_x, train_y = self.x.iloc[train_idx], self.y.iloc[train_idx]
            valid_x, valid_y = self.x.iloc[valid_idx], self.y.iloc[valid_idx]

            model.fit(train_x, train_y, eval_set=(valid_x, valid_y), use_best_model=True)
            valid_predict = model.predict(valid_x)
            model.save_model(os.path.join(model_path, "trained_model_" + (str(index))))

            print(index, metrics.mean_squared_error(valid_y, valid_predict))
            oof[valid_idx] = valid_predict

        mean_squared_error = metrics.mean_squared_error(self.y, oof)
        print("mean_squared_error: ", mean_squared_error)
        return mean_squared_error

if __name__ == "__main__":
    # data = pd.read_csv("train.csv")
    # print(data.info())
    pass