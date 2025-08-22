# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 13:48:49 2025

@author: lch99
"""

# 导入库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 准备数据（以鸢尾花数据集为例）
data = load_iris()
X = data.data  # 特征（花萼长度、宽度等）
y = data.target  # 目标类别（3种鸢尾花）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 构建决策树模型（CART算法，默认基尼指数）
clf = DecisionTreeClassifier(
    criterion='gini',  # 特征选择指标（'gini'或'entropy'）
    max_depth=5,  # 预剪枝：限制树的最大深度
    min_samples_leaf=5  # 预剪枝：叶子节点最小样本数
)

# 3. 训练模型
clf.fit(X_train, y_train)

# 4. 预测与评估
y_pred = clf.predict(X_test)
print("准确率：", accuracy_score(y_test, y_pred))  # 输出预测准确率

# 5. 可视化决策树


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,  # 按类别填充颜色
    rounded=True  # 圆角边框
)
plt.show()
