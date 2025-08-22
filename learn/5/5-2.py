# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:34:46 2025

@author: lch99
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 加载数据集
iris = load_iris()
X = iris.data[:, :2]  # 只使用前两个特征
y = (iris.target != 0) * 1  # 将目标转化为二分类问题
print(X.shape)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")
'''
# 混淆矩阵 
# [[TN, FP],
#[FN, TP]]
TN：真实为 0，预测为 0（真负例）；
FP：真实为 0，预测为 1（假正例）；
FN：真实为 1，预测为 0（假负例）；
TP：真实为 1，预测为 1（真正例）。
'''
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(conf_matrix)
'''
# 分类报告
          precision(精准率)   recall（召回）  f1-score   support（支持度）

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        26

    accuracy（准确率）                  1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
'''
class_report = classification_report(y_test, y_pred)
print("分类报告:")
print(class_report)