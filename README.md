# uselss_stock
使用 tsfresh + catboost 的股票预测


## 每周预测推荐（2023-03-15）
  股票代码     | 预测五日（%）         
-----------|--------------
|sh.600339	|4.321787999|
|sz.001236	|3.428373024|
|sh.603090	|3.245037789|




## 更新日志
- 2023-02-28：修正特征提取，不再使用采样中相关的取交集，使用catboost的特征重要度来选择top200的特征进行生成训练和预测。
- 2023-01-15：修正测试集合为未见过的股票集合，取消5折交叉验证。
- 2023-01-13：删除创业板和st股票，因为创业板每天20%而st5%与常规的10%不同这里直接剔除。
- 2023-01-12：修复bug，特征提取的时候压缩每支股票的样本，使用随机采样，这样可以对更多的股票进行特征相关性分析。