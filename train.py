import argparse
import yaml
import os
import pandas as pd

from downloader import download_all
from data_process import data_process
from Trainer import Trainer


def decode_configer(path="config.yaml"):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    # init dirs
    for dir in config['path'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)
    return config


if __name__ == "__main__":
    # 参数备用
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # 参数读取
    config = decode_configer()
    # 下载数据 增量下载
    download_all(config['path'])
    print("原始数据下载完成！")
    # 数据预处理
    data_process(config)
    # 读取处理好的数据作为输入
    paths = config['path']
    trian_path = os.path.join(paths['processed_data'], "train.csv")
    data = pd.read_csv(trian_path)
    with open("selected_features.txt", "r") as f:
        selected_features = [item.replace('\n', '') for item in f.readlines()]
    data = data[selected_features + ['target']]
    print(data.info())
    data = data.dropna(axis=0, subset = ["target"])

    features = [x for x in data.columns if x not in ['target']]
    train_x = data[features].copy()
    train_y = data['target']

    # 训练
    trainer = Trainer(train_x=train_x, train_y=train_y)
    trainer.catboost_train(cv=5, model_path=paths['trained_model'])

    