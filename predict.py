from catboost import CatBoostRegressor
import os
import pandas as pd
import matplotlib.pylab as plt
from tsfresh import extract_features
import tsfresh
from tqdm import tqdm
import yaml
import warnings
warnings.filterwarnings('ignore')

def decode_configer(path="config.yaml"):
    with open(path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # init dirs
    for dir in config['path'].values():
        if not os.path.exists(dir):
            os.makedirs(dir)
    return config

def get_now_sample(config):
    paths = config['path']
    ml_parameters = config['ml_parameters']
    # 参数
    look_back = ml_parameters['look_back']
    features = ml_parameters['original_features']
    n_job = ml_parameters['n_job']
    data_dir = paths['original_data']

    collected_x = pd.DataFrame()
    codes = []
    for p in tqdm(os.listdir(data_dir)):
        df = pd.read_csv(os.path.join(data_dir, p))
        # 固有属性
        code_name = df['code'].iloc[0]
        # 要部分特征
        df['pctRange'] = (df['high'] - df['low']) / df['low']
        # 我发现不适用当日的成交价似乎更好一点
        df = df[features]
        x = df.iloc[-look_back:]
        x['code_s'] = code_name
        collected_x= pd.concat([collected_x, x])
        codes.append(code_name)

    with open(os.path.join(paths['config'], "selected_features.txt"), "r") as f:
        selected_features = [item.replace('\n', '') for item in f.readlines()]
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(
        selected_features)
    collected_x = collected_x.fillna(0)
    featured_x = extract_features(timeseries_container=collected_x,
                kind_to_fc_parameters=kind_to_fc_parameters,
                column_id='code_s',
                n_jobs=n_job,
                disable_progressbar=True
                )
    return featured_x

def predict(data, paths):
    # 参数
    model_dir = paths['trained_model']

    # 剔除代码对应的列
    features = [x for x in data.columns if x not in ['code']]
    X = data[features].copy()

    model_list = []
    res = []
    for p in os.listdir(model_dir):
        model = CatBoostRegressor(task_type="GPU")
        model.load_model(os.path.join(model_dir, p))
        model_list.append(model)

    res = None
    for model in model_list:
        predict = model.predict(X)
        if res is None:
            res = predict
        else:
            res += predict

    return res / len(model_list)


if __name__ == "__main__":
    # data = pd.read_csv("./valid.csv")
    # res = predict("./trained_model/", data)
    # index = np.argsort(-res)[:20]
    # k = 0
    # for i in index:
    #     if np.array(data['target'])[i] > 3:
    #         k += 1
    # print(k / 20 * 100)
    # plt.plot(res[index])
    # plt.plot(list(np.array(data['target'])[index]))
    # plt.show()
    config = decode_configer()
    data = get_now_sample(config)
    res = predict(data, config['path'])
    res = pd.DataFrame(res)
    res.index = data.index
    res.to_csv("result.csv")
