# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from boost.gradient_boosting import GradientBoostingRegressor as FWGBR


data_path = "./dataset"
train = pd.read_csv("{}/d_train_20180102.csv".format(data_path), encoding="gbk")
test = pd.read_csv("{}/d_test_A_20180102.csv".format(data_path), encoding="gbk")
train.drop([4160], inplace=True)
train["中淋"] = train["中性粒细胞%"].values / train["淋巴细胞%"]
test["中淋"] = test["中性粒细胞%"].values / test["淋巴细胞%"]
train["性别"] = train["性别"].map({"男": 1, "女": 0, "??": 1})
test["性别"] = test["性别"].map({"男": 1, "女": 0, "??": 1})
train_feat = train.drop(["体检日期", "血糖"], axis=1)
test_feat = test.drop(["体检日期"], axis=1)

train_feat.drop(["id"], axis=1, inplace=True)
test_feat.drop(["id"], axis=1, inplace=True)
train_feat.drop(["乙肝表面抗原", "乙肝表面抗体", "乙肝e抗原", "乙肝e抗体", "乙肝核心抗体"], axis=1, inplace=True)
test_feat.drop(["乙肝表面抗原", "乙肝表面抗体", "乙肝e抗原", "乙肝e抗体", "乙肝核心抗体"], axis=1, inplace=True)

train_feat.fillna(0, inplace=True)
test_feat.fillna(0, inplace=True)
print(train_feat.shape, test_feat.shape)

Y = train["血糖"].values
print(Y)
train_feat = np.array(train_feat)
test_feat = np.array(test_feat)

C4 = np.zeros_like(Y)
a4, b4, c4, d4 = Y < 3.9, (Y >= 3.9) & (Y <= 6.1), (Y > 6.1) & (Y < 10), Y >= 10
C4[a4], C4[b4], C4[c4], C4[d4] = 0, 1, 2, 3

# setup parameters
SK = 5
random_state = 520

C = C4
unique_C = np.unique(C)
skf = StratifiedKFold(n_splits=SK, random_state=random_state, shuffle=True)


train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], SK))
for index, (train_index, test_index) in enumerate(skf.split(train_feat, C)):
    train_X, test_X = train_feat[train_index], train_feat[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]
    train_C, test_C = C[train_index], C[test_index]
    abc = FWGBR(n_estimators=200, learning_rate=0.01)
    abc.fit(train_X, train_Y)

    train_preds[test_index] += abc.predict(test_X)
    test_preds[:, index] = abc.predict(test_feat)

print("线下得分：    {}".format(mean_squared_error(Y, train_preds) * 0.5))
t = pd.DataFrame({"pred": train_preds})
t.to_csv(
    "./sub/train_pred.csv",
    header=None,
    index=False,
    float_format="%.4f"
)
submission = pd.DataFrame({"pred": test_preds.mean(axis=1)})
submission.to_csv(
    "./sub/sub{}.csv".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
    header=None,
    index=False,
    float_format="%.4f"
)
