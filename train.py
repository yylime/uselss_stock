import argparse
import yaml
import os
import pandas as pd

from downloader import download_all
from data_process import data_process
from Trainer import Trainer


def decode_configer(path="config.yaml"):
    with open(path, 'r', encoding="utf-8") as f:
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
    data_process(config, config['featured_extract'])
    # 方便调试
    if config['train']:
        # 读取处理好的数据作为输入
        paths = config['path']
        with open(os.path.join(paths['config'], "selected_features.txt"), "r") as f:
            
            selected_features = [item.replace('\n', '') for item in f.readlines()]

        trian_path = os.path.join(paths['processed_data'], "train.csv")
        train_data = pd.read_csv(trian_path)
        train_data = train_data[selected_features + ['target']]
        train_data = train_data.dropna(axis=0, subset = ["target"])
        # valid
        valid_path = os.path.join(paths['processed_data'], "valid.csv")
        valid_data = pd.read_csv(valid_path)
        valid_data = valid_data.dropna(axis=0, subset = ["target"])

        features = [x for x in train_data.columns if x not in ['target']]

        train_x = train_data[features].copy()
        train_y = train_data['target']
        valid_x = valid_data[features].copy()
        valid_y = valid_data['target']

        # 训练
        trainer = Trainer(train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y)
        trainer.catboost_train_with_valid(paths['trained_model'])
