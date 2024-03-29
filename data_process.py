#!/usr/bin/env python
# coding: utf-8

# ## 特征需要
# ### 最为重要的是使用多少天预测多少天
# - 样本稀疏化
# - 以15天为基准，采用最近的5天的数据波动情况来做预测
# ### 特征值5天的
# - 5日内（部分特征）的 均值、方差、最值、振幅、skew
# - 前一1天的量在之前五天内的情况QAQ
# # 2022-10-19 使用已有的构造
# https://tsfresh.readthedocs.io/en/latest/text/quick_start.html
#

from collections import Counter
import pandas as pd
from tsfresh import extract_features, extract_relevant_features
import os
import random
import warnings
from multiprocessing import Pool
from tqdm import tqdm
import tsfresh
from random import sample
import talib
import talib.abstract as ta
import numpy as np

warnings.filterwarnings('ignore')


## 增加新的常用的指标
def add_indectors(df: pd.DataFrame):
    ## add macd and kdj
    # Calculate the MACD indicator
    df = df.fillna(0)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

    # Calculate the KDJ indicator
    high = df['high'].rolling(9).max()
    low = df['low'].rolling(9).min()
    rsv = (df['close'] - low) / (high - low) * 100
    rsv = rsv.fillna(0)
    df['k'] = talib.SMA(rsv, timeperiod=3)
    df['d'] = talib.SMA(df['k'], timeperiod=3)
    df['j'] = 3 * df['k'] - 2 * df['d']
    df['kMj'] = df['k'] * df['j'] / (df['k'] + df['j'])
    df['kdj'] = (df['k'] >= df['d']) * 4 +  (df['d'] >= df['j']) * 2  + (df['j'] >= df['k'])
    rsi_period = 14
    df['rsi'] = ta.RSI(df, timeperiod=rsi_period, price='close')
    # SMA features
    df['ma5'] = ta.SMA(df['close'], timeperiod=5)
    df['ma10'] = ta.SMA(df['close'], timeperiod=10)
    df['ma20'] = ta.SMA(df['close'], timeperiod=20)
    df["ma5_close"] = df['ma5'] - df['close']
    df["ma10_close"] = df['ma10'] - df['close']
    df["ma20_close"] = df['ma20'] - df['close']
    df['ma5_ma20'] = df['ma5'] - df['ma20']
    # obv
    df['obv'] = talib.OBV(df['close'], df['volume'])
    df['atr']=ta.ATR(df.high, df.low, df.close, timeperiod=14)
    df['natr']=ta.NATR(df.high, df.low, df.close, timeperiod=14)

    # drop before nan
    df = df.iloc[20:]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df



def pre_proceeDf(df, features):
    # 对y的缺失值填补
    df['pctChg'] = df['pctChg'].fillna(0)
    # 要部分特征
    df['pctRange'] = (df['high'] - df['low']) / df['low']
    # 增加平均成交以及今日涨跌与平均成交的偏离度
    df['mean_deal'] = df['amount'] / (df['volume'] + 1e-8)
    df['bd'] = (df['close'] - df['preclose']) / (df['mean_deal'] + 1e-8)
    # 加入close似乎有负面的影响 2022-01-12
    df = add_indectors(df)
    df = df[features]
    return df


# 根据look_back 和 look_up 对时间序列进行切片
def get_samples(path, features:list, look_back=15, look_up=5, ts_rate=1):
    df = pd.read_csv(path)
    # 固有属性
    code_name = df['code'].iloc[0]
    # 手动特征
    df = pre_proceeDf(df, features)

    #便于后续的采样，ret_x使用list来存储
    ret_x, ret_y = [], pd.Series()
    n = len(df)
    for i in range(n - look_back - look_up):
        x = df.iloc[i:i + look_back]
        x['code_s'] = code_name + '_' + str(i)
        # y = sum(df['pctChg'].iloc[i + look_back:i + look_back + look_up])
        # 上述的计算方法是错误的，百分比的叠加不能正确反映出真实的涨跌幅度，应该使用 close - pre_close / pre_close * 100 计算
        y = (df['close'].iloc[i + look_back + look_up] - df['close'].iloc[i + look_back - 1]) / df['close'].iloc[i + look_back - 1]
        # 对时间样本进行随机采样
        if random.random() > ts_rate:
            continue
        ret_x.append(x)
        ret_y[code_name + '_' + str(i)] = y
    return ret_x, ret_y


# 特征筛选
def featured_select(path_list, paths, config):
    # 参数
    sample_num, sample_batch_size, sample_rate, sample_ts_num = config['sample_num'], config[
        'sample_batch_size'], config['sample_rate'], config['sample_ts_num']
    look_back, look_up = config['look_back'], config['look_up']
    n_job = config['n_job']
    ts_rate = config['ts_rate']
    # 初始化
    sample_list = sample(path_list, sample_num)
    collected_x = pd.DataFrame()
    collected_y = pd.Series()
    all_features = []
    # tsfresh 自带多进程，无需额外开启
    for idx, p in enumerate(tqdm(sample_list), start=1):
        x, y = get_samples(p, config['original_features'], look_back, look_up, ts_rate)
        sample_ts_set = sample(range(len(x)), min(sample_ts_num, len(x)))
        sample_x = [x[i] for i in sample_ts_set]
        collected_x = pd.concat([collected_x] + sample_x)
        collected_y = pd.concat([collected_y, y.iloc[sample_ts_set]])
        if idx % sample_batch_size == 0:
            # 后续考虑丢弃缺失值，目前暂定fillna
            y = pd.Series(collected_y).fillna(0)
            collected_x = collected_x.fillna(0)
            # 提取相关特征
            featured_x = extract_relevant_features(
                timeseries_container=collected_x,
                column_id='code_s',
                y=y,
                n_jobs=n_job,
                disable_progressbar=True)
            all_features.extend(list(featured_x.columns))
            # download sample
            featured_x['y'] = y
            featured_x.to_csv(
                os.path.join(paths['selected_data'],
                             f"feature_select_batc_{idx}.csv"))
            collected_x = pd.DataFrame()
            collected_y = pd.Series()

    # 特征数量选择
    all_features = [f + '\n' for f in all_features]
    counter = Counter(all_features)
    rate = int(sample_num / sample_batch_size * sample_rate)
    selected_features = []
    for f, cnt in counter.items():
        if cnt >= rate:
            selected_features.append(f)
    with open(os.path.join(paths['config'], "all_samples_features.txt"), 'w') as f:
        f.writelines(all_features)
    with open(os.path.join(paths['config'], "selected_features.txt"), 'w') as f:
        f.writelines(selected_features)
    return selected_features


# # 测试集和验证集样本构建
def merge_samples(path_list, config, f="train"):
    # 参数设置
    paths = config['path']
    ml_parameters = config['ml_parameters']
    look_back, look_up = ml_parameters['look_back'], ml_parameters['look_up']
    n_job = ml_parameters['n_job']
    train_batch = ml_parameters['train_batch']
    if f == "train":
        ts_rate = ml_parameters['ts_rate']
    else:
        ts_rate = 1
    # 数据返回
    collected_x, collected_y = pd.DataFrame(), pd.Series()
    ret_df = pd.DataFrame()
    num = 0
    # 特征筛选 检查是否做过
    with open(os.path.join(paths['config'], "selected_features.txt"), "r") as f:
        selected_features = [item.replace('\n', '') for item in f.readlines()]
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(
        selected_features)

    for num, p in enumerate(tqdm(path_list), start=1):
        x, y = get_samples(p, ml_parameters['original_features'], look_back, look_up, ts_rate)
        collected_x = pd.concat([collected_x] + x)
        collected_y = pd.concat([collected_y, y])
        collected_x = collected_x.fillna(0)

        if num % train_batch == 0 or num == len(path_list):
            featured_x = extract_features(
                timeseries_container=collected_x,
                kind_to_fc_parameters=kind_to_fc_parameters,
                column_id='code_s',
                n_jobs=n_job,
                disable_progressbar=True)
            featured_x['target'] = collected_y
            ret_df = ret_df.append(featured_x)
            collected_x = pd.DataFrame()
            collected_y = pd.Series()

    return ret_df


def data_process(config):
    paths = config['path']
    ml_parameters = config['ml_parameters']

    root_path = paths['original_data']
    all_paths = [os.path.join(root_path, p) for p in os.listdir(root_path)]
    # 随机
    random.seed(823)
    r_all_paths = random.sample(all_paths, len(all_paths))
    n_split = int(len(r_all_paths) * ml_parameters['train_split'])
    train_paths, vaild_paths = r_all_paths[:n_split], r_all_paths[n_split:]
    # 特征选择，这里采用的是随机采样QAQ
    if config['featured_extract']:
        features = featured_select(train_paths, paths, ml_parameters)
        print("特征提取完毕，请在config目录下查看")
        return "只进行了特征提取"
    else:
        # 数据处理
        train_data = merge_samples(train_paths, config, f="train")
        train_data.to_csv(os.path.join(paths['processed_data'], 'train.csv'),
                        encoding='utf-8',
                        index=False)
        valid_data = merge_samples(vaild_paths, config, f="valid")
        valid_data.to_csv(os.path.join(paths['processed_data'], 'valid.csv'),
                        encoding='utf-8',
                        index=False)


if __name__ == "__main__":
    path = r"data\all_data\sz.003042.csv"
    data = pd.read_csv(path)
    print(data.head())
    data = add_indectors(data)
    print(data.iloc[100:108])
