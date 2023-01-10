# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time      : 2021/8/21 18:02
# Author    : yyl
# email     : 844202100@qq.com
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta, date
import os
from tqdm import tqdm
from multiprocessing import Pool

bs.login()
def download_one(code, start_date='2019-01-01', end_date=date.today().strftime(r"%Y-%m-%d"), path="all_data"):
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, code + ".csv")
    if os.path.exists(path):
        data = pd.read_csv(path, index_col=None)
        start_date = data['date'].iloc[-1]
    else:
        data = pd.DataFrame()
    # data = pd.DataFrame()
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(
        code,
        # 指标的中文对应可以参考 http://baostock.com/baostock/index.php/A%E8%82%A1K%E7%BA%BF%E6%95%B0%E6%8D%AE
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
        "pctChg,peTTM,psTTM,pcfNcfTTM",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3")
    data = data.append(rs.get_data())
    data.to_csv(path, encoding='utf-8', index=False)


def download_online_codes():
    stock_rs = bs.query_all_stock('2022-12-27')
    result = stock_rs.get_data()
    ret = []
    for i in range(len(result)):
        code, status, code_name  = result.iloc[i]
        if 'bj' in code: continue
        ## 筛选出还在上市的股票
        if status == "1" and len(code_name) > 0:
            rs = bs.query_stock_basic(code)
            is_online_stock = True
            for key, value in zip(rs.fields, rs.data[0]):
                if key == 'type' and value != '1':
                    is_online_stock = False
                elif key == 'status' and value != '1':
                    is_online_stock = False
            if is_online_stock:
                ret.append(",".join(rs.data[0]) + '\n')
    with open('code.txt', 'w') as f:
        f.writelines(ret)
    return ret

# 这里使用多线程下载
def download_all(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    codes = [line.split(',')[0] for line in lines]
    pool = Pool(12)
    pool.map(download_one, codes)
    return codes

if __name__ == '__main__':

    # 第一次运行需要加入
    download_all("config/code.txt")
    bs.logout()
