#!/usr/bin/env python
# coding: utf-8

# ## 特征需要
#
# ### 最为重要的是使用多少天预测多少天
# - 样本稀疏化
# - 以15天为基准，采用最近的5天的数据波动情况来做预测
# -
#
# ### 特征值5天的
# - 5日内（部分特征）的 均值、方差、最值、振幅、skew
# - 前一1天的量在之前五天内的情况QAQ
#
# # 2022-10-19 使用已有的构造
# https://tsfresh.readthedocs.io/en/latest/text/quick_start.html
#

import pandas as pd
from tsfresh import extract_features, extract_relevant_features
import os
import random
import warnings
from multiprocessing import Pool
from tqdm import tqdm
from tsfresh.utilities.distribution import MultiprocessingDistributor
import tsfresh
from random import sample
warnings.filterwarnings('ignore')

def get_samples(path, look_back=15, look_up=5):
    df = pd.read_csv(path)
    # 固有属性
    code_name = df['code'].iloc[0]
    # 要部分特征
    df['pctRange'] = (df['high'] - df['low']) / df['low']
    df = df[['close', 'volume','turn', 'pctChg', 'pctRange', 'peTTM','psTTM','pcfNcfTTM']]

    ret_x, ret_y = pd.DataFrame(), {}
    n = len(df)
    for i in range(n - look_back - look_up):
        x = df.iloc[i:i + look_back]
        x['code_s'] = code_name + '_' + str(i)
        y = sum(df['pctChg'].iloc[i + look_back:i + look_back + look_up])
        ret_x = ret_x.append(x)
        ret_y[code_name + '_' + str(i)] = y
    # ret_x['isSt'] = isSt
    return ret_x, ret_y

# 特征筛选
def featured_select(path_list):
    sample_list = sample(path_list, 100)
    collected_x = pd.DataFrame()
    collected_y = {}
    num = 0
    all_features = []
    for p in tqdm(sample_list):
        x, y = get_samples(p)
        collected_x = collected_x.append(x)
        collected_y.update(y)

        num += 1
        if num % 10 == 0:
            y = pd.Series(collected_y).fillna(0)
            collected_x = collected_x.fillna(0)
            featured_x = extract_relevant_features(timeseries_container=collected_x,
                                column_id='code_s',
                                y=y,
                                n_jobs=4,
                                disable_progressbar=True
                                )
            all_features.append(list(featured_x.columns))
            collected_x = pd.DataFrame()
            collected_y = {}

    with open ('f.txt', 'w') as f:
        temp = sum(all_features, [])
        temp = [i + '\n' for i in temp]
        f.writelines(temp)
    ret = []
    for item in sum(all_features, []):
        p = 0
        for nt in all_features:
            for ntitem in nt:
                if item == ntitem:
                    p += 1
        if p >= 5:
            ret.append(item)
    return ret


# # 测试集和验证集样本构建
def merge_samples(path_list):
    collected_x = pd.DataFrame()
    collected_y = {}
    ret_df = pd.DataFrame()
    num = 0
    # 特征筛选
    with open("selected_features.txt", "r") as f:
        selected_features = [item.replace('\n', '') for item in f.readlines()]
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(
        selected_features)

    for p in tqdm(path_list):
        x, y = get_samples(p)
        collected_x = collected_x.append(x)
        collected_y.update(y)
        collected_x = collected_x.fillna(0.)

        num += 1
        if num % 200 == 0 or num == len(path_list):
            featured_x = extract_features(timeseries_container=collected_x,
                            kind_to_fc_parameters=kind_to_fc_parameters,
                            column_id='code_s',
                            n_jobs=3,
                            disable_progressbar=True
                            )
            featured_x['target'] = collected_y
            ret_df = ret_df.append(featured_x)

            collected_x = pd.DataFrame()
            collected_y = {}

    return ret_df

if __name__ == "__main__":
    root_path = 'all_data'
    all_paths = [os.path.join(root_path, p) for p in os.listdir(root_path)]
    # 随机
    random.seed(823)
    r_all_paths = random.sample(all_paths, len(all_paths))
    n_split = int(len(r_all_paths) * 0.9)
    train_paths, vaild_paths = r_all_paths[:n_split], r_all_paths[n_split:]

    # 特征选择，这里采用的是随机采样QAQ
    # features = featured_select(train_paths)
    # print(features)

    train_data = merge_samples(train_paths)
    train_data.to_csv('train.csv', encoding='utf-8', index=False)

    valid_data = merge_samples(vaild_paths)
    valid_data.to_csv('valid.csv', encoding='utf-8', index=False)
