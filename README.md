# uselss_stock
使用 tsfresh + catboost 的股票预测


## 每周预测推荐（2023-01-16）
  股票代码     | 预测五日（%）         
-----------|--------------
sz.002195 |	6.9067276
sh.605289	| 4.123113691
sz.001896	| 4.028459705



## 更新日志
- 2023-01-15：修正测试集合为未见过的股票集合，取消5折交叉验证。
- 2023-01-13：删除创业板和st股票，因为创业板每天20%而st5%与常规的10%不同这里直接剔除。
- 2023-01-12：修复bug，特征提取的时候压缩每支股票的样本，使用随机采样，这样可以对更多的股票进行特征相关性分析。