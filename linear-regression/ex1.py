#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: dating.py
@Author: yew1eb
@Date: 2015/12/20
"""
'''
1. 使用线性回归模型预测房价
2. 预测下周有多少观众
3. 替换数据中缺失值
'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model  # 导入线性回归

# pd读取csv, return x,y列表


def get_data(file_name):
    data = pd.read_csv(file_name)
    X_param = []
    Y_param = []
    for single_square_feet, single_price_value in zip(data['square_feet'], data['price']):
        X_param.append([float(single_square_feet)])
        Y_param.append(float(single_price_value))
    return X_param, Y_param


# 输入为X_parameters、Y_parameter和你要预测的平方英尺值，
# 返回θ0、θ1和预测出的价格值。
def linear_model_main(X_param, Y_param, predict_value):
    # 构建线性回归模型
    regr = linear_model.LinearRegression()  # 使用线性回归
    regr.fit(X_param, Y_param)  # 对输入和输出进行一次fit，训练出一个模型
    # 预测
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_  # 系数矩阵
    predictions['predicted_value'] = predict_outcome
    return predictions


X_param, Y_param = get_data("ex1_data.csv")
predict_value = 700
result = linear_model_main(X_param, Y_param, predict_value)
print("Intercept value: ", result['intercept'])
print("coefficient: ", result['coefficient'])
print("Predicted value: ", result['predicted_value'])


# 显示拟合直线
def show_linear_line(X_param, Y_param):
    regr = linear_model.LinearRegression()
    regr.fit(X_param, Y_param)

    plt.scatter(X_param, Y_param, color='blue')
    plt.plot(X_param, regr.predict(X_param), color='red', linewidth=2)
    plt.xticks(())
    plt.yticks(())
    plt.show()

# show_linear_line(X_param, Y_param)

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
print X.head()
y = data['Sales']
# print y
lr = linear_model.LinearRegression()
lr.fit(X, y)
print lr.intercept_
print lr.coef_
print zip(feature_cols, lr.coef_)
# lr.predict([])

# predict_value = 700
# result = linear_model_main(X_param, Y_param, predict_value)
# print("Intercept value: ", result['intercept'])
# print("coefficient: ", result['coefficient'])
# print("Predicted value: ", result['predicted_value'][0])


# # 显示拟合直线
# def show_linear_line(X_param, Y_param):
#     regr = linear_model.LinearRegression()
#     regr.fit(X_param, Y_param)

#     plt.scatter(X_param, Y_param, color='blue')
#     plt.plot(X_param, regr.predict(X_param), color='red', linewidth=2)
#     plt.xticks(())
#     plt.yticks(())
#     plt.show()

# show_linear_line(X_param, Y_param)
