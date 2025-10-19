import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 加载鸢尾花数据集（经典的多分类问题）
iris = load_iris()
X, y = iris.data, iris.target

print(f"数据集信息:")
print(f"- 样本数量: {X.shape[0]}")
print(f"- 特征数量: {X.shape[1]}")
print(f"- 类别数量: {len(np.unique(y))}")
print(f"- 类别名称: {iris.target_names}")

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 使用多项逻辑回归（默认方法）
model_multinomial = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42
)
model_multinomial.fit(X_train, y_train)

# 使用一对多方法
model_ovr = LogisticRegression(
    multi_class="ovr", solver="liblinear", max_iter=1000, random_state=42
)
model_ovr.fit(X_train, y_train)

# 评估模型
y_pred_multinomial = model_multinomial.predict(X_test)
y_pred_ovr = model_ovr.predict(X_test)

print("\n多项逻辑回归性能:")
print(classification_report(y_test, y_pred_multinomial, target_names=iris.target_names))

print("\n一对多方法性能:")
print(classification_report(y_test, y_pred_ovr, target_names=iris.target_names))

# 获取概率预测
probabilities_multinomial = model_multinomial.predict_proba(X_test)
probabilities_ovr = model_ovr.predict_proba(X_test)

print(f"\n概率一致性检查:")
print(f"多项逻辑回归概率和: {np.sum(probabilities_multinomial, axis=1)}")
print(f"一对多方法概率和: {np.sum(probabilities_ovr, axis=1)}")
