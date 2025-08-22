# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 21:56:17 2025
@author: lch99
功能：信用风险建模分析，包含数据生成、预处理和三个级联模型
      1. 线性回归预测月收入
      2. 逻辑回归预测是否逾期
      3. 决策树预测信用等级
"""

# 导入必要的库
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.preprocessing import StandardScaler  # 特征标准化
from sklearn.impute import SimpleImputer  # 缺失值填充
# 导入模型：线性回归、逻辑回归、决策树分类器
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# 导入评估指标：均方误差、准确率、分类报告
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.tree import plot_tree  # 可视化决策树
import matplotlib.pyplot as plt  # 绘图


# ---------------------- 1. 数据生成与预处理 ----------------------
# 设置随机种子，保证结果可复现
np.random.seed(42)
# 生成1000条样本
n = 1000
# 构建基础特征字典
data = {
    "age": np.random.randint(18, 65, n),  # 年龄：18-64岁的随机整数
    "work_year": np.random.randint(0, 40, n),  # 工作年限：0-39年的随机整数
    # 教育程度：0=高中及以下，1=本科，2=研究生；按比例[30%,50%,20%]生成
    "education": np.random.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2]),
    "debt_ratio": np.random.uniform(0.1, 0.8, n),  # 债务比率：0.1-0.8的随机浮点数
    "credit_card_num": np.random.randint(1, 10, n)  # 信用卡数量：1-9张的随机整数
}

# 生成月收入（受多个因素影响的模拟数据）
# 基础公式：3000起薪 + 工作年限*800 + 教育程度*2000 + 年龄*50 + 随机波动
data["monthly_income"] = 3000 + data["work_year"]*800 + data["education"]*2000 + data["age"]*50 + np.random.normal(0, 1000, n)

# 生成是否逾期的标签（0=未逾期，1=逾期）
# 第一步：用sigmoid函数计算逾期概率（将线性组合压缩到0-1之间）
# 线性组合公式：-0.0005*月收入（收入越高，逾期概率越低） + 3*债务比率（债务越高，逾期概率越高） -0.01*年龄（年龄越大，逾期概率越低）
overdue_prob = 1 / (1 + np.exp(-( -0.0005*data["monthly_income"] + 3*data["debt_ratio"] - 0.01*data["age"] )))
# 第二步：根据概率用伯努利分布生成0/1标签（1表示逾期，概率由overdue_prob决定）
data["is_overdue"] = np.random.binomial(1, overdue_prob)

# 生成信用等级（A/B/C/D，模拟实际信用评分场景）
# 第一步：计算评分（受收入、逾期状态、教育程度、债务比率影响）
rating_score = (data["monthly_income"]/100) - (data["is_overdue"]*500) + (data["education"]*200) - (data["debt_ratio"]*1000)
# 第二步：根据评分区间划分等级（A最高，D最低）
data["credit_rating"] = pd.cut(rating_score, bins=[-np.inf, 30, 60, 90, np.inf], labels=["D", "C", "B", "A"])

# 将字典转换为DataFrame（表格形式）
df = pd.DataFrame(data)

# 划分训练集和测试集（80%训练，20%测试，固定随机种子保证划分一致）
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 模拟10%的月收入数据缺失（现实中数据常存在缺失）
train_df.loc[train_df.sample(frac=0.1).index, "monthly_income"] = np.nan  # 随机选择10%的行设为缺失
# 用中位数填充缺失值（中位数对异常值不敏感，适合收入这类可能有极端值的数据）
imputer = SimpleImputer(strategy="median")  # 创建填充器，策略为"中位数"
# 注意：fit_transform需要传入二维数组（所以用[["monthly_income"]]而不是"monthly_income"）
train_df["monthly_income"] = imputer.fit_transform(train_df[["monthly_income"]])
test_df["monthly_income"] = imputer.transform(test_df[["monthly_income"]])  # 测试集用训练集的填充规则

# 特征标准化（线性模型对特征尺度敏感，标准化后收敛更快且系数可比较）
scaler = StandardScaler()  # 创建标准化器（将特征转换为均值0、标准差1）
# 需要标准化的数值特征列表
num_features = ["age", "work_year", "debt_ratio", "credit_card_num", "monthly_income"]
# 训练集拟合并转换
train_df[num_features] = scaler.fit_transform(train_df[num_features])
# 测试集仅转换（用训练集的均值和标准差，避免数据泄露）
test_df[num_features] = scaler.transform(test_df[num_features])


# ---------------------- 2. 线性回归：预测月收入 ----------------------
# 选择用于预测月收入的特征：年龄、工作年限、教育程度、信用卡数量
lr_features = ["age", "work_year", "education", "credit_card_num"]
# 构建训练数据（特征X和目标y）
X_lr_train = train_df[lr_features]  # 训练特征
y_lr_train = train_df["monthly_income"]  # 训练目标（月收入）
X_lr_test = test_df[lr_features]  # 测试特征
y_lr_test = test_df["monthly_income"]  # 测试目标

# 初始化并训练线性回归模型
lr = LinearRegression()  # 线性回归模型（默认参数）
lr.fit(X_lr_train, y_lr_train)  # 拟合训练数据

# 用训练好的模型在测试集上预测
y_lr_pred = lr.predict(X_lr_test)
# 评估预测效果：计算均方误差（MSE），值越小说明预测越准确
print(f"线性回归月收入预测 MSE：{mean_squared_error(y_lr_test, y_lr_pred):.4f}")


# ---------------------- 3. 逻辑回归：预测是否逾期 ----------------------
# 选择用于预测逾期的特征：债务比率、年龄
logit_features = ["debt_ratio", "age"]
# 将线性回归预测的月收入作为新特征加入（用模型预测的收入代替原始收入，模拟实际中收入可能难以准确获取的场景）
train_df["pred_income"] = lr.predict(train_df[lr_features])  # 训练集用模型预测收入
test_df["pred_income"] = y_lr_pred  # 测试集直接用之前的预测结果

# 构建逻辑回归的训练数据（包含原始特征和新特征）
X_logit_train = train_df[logit_features + ["pred_income"]]  # 特征：债务比率 + 年龄 + 预测收入
y_logit_train = train_df["is_overdue"]  # 目标：是否逾期（0/1）
X_logit_test = test_df[logit_features + ["pred_income"]]  # 测试特征
y_logit_test = test_df["is_overdue"]  # 测试目标

# 初始化并训练逻辑回归模型
# class_weight="balanced"：自动调整类别权重，解决可能的样本不平衡问题（逾期样本可能较少）
logit = LogisticRegression(class_weight="balanced")
logit.fit(X_logit_train, y_logit_train)  # 拟合训练数据

# 在测试集上预测
y_logit_pred = logit.predict(X_logit_test)
# 评估效果：准确率（整体正确率）和分类报告（包含精确率、召回率、F1值）
print(f"逻辑回归逾期预测准确率：{accuracy_score(y_logit_test, y_logit_pred):.4f}")
print("逻辑回归分类报告：")
print(classification_report(y_logit_test, y_logit_pred))  # 详细评估每个类别的表现


# ---------------------- 4. 决策树：预测信用等级 ----------------------
# 选择用于预测信用等级的特征：教育程度、信用卡数量
tree_features = ["education", "credit_card_num"]
# 将逻辑回归预测的逾期结果作为新特征加入
train_df["pred_overdue"] = logit.predict(train_df[logit_features + ["pred_income"]])  # 训练集预测逾期
test_df["pred_overdue"] = y_logit_pred  # 测试集用之前的预测结果

# 构建决策树的训练数据（组合原始特征和前两个模型的预测结果）
X_tree_train = train_df[tree_features + ["pred_income", "pred_overdue"]]  # 特征：教育程度 + 信用卡数量 + 预测收入 + 预测逾期
y_tree_train = train_df["credit_rating"]  # 目标：信用等级（A/B/C/D）
X_tree_test = test_df[tree_features + ["pred_income", "pred_overdue"]]  # 测试特征
y_tree_test = test_df["credit_rating"]  # 测试目标

# 初始化并训练决策树模型
# max_depth=5：限制树深度，避免过拟合（决策树容易过度复杂而记住训练数据）
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_tree_train, y_tree_train)  # 拟合训练数据

# 在测试集上预测
y_tree_pred = tree.predict(X_tree_test)
# 评估效果
print(f"决策树信用等级预测准确率：{accuracy_score(y_tree_test, y_tree_pred):.4f}")
print("决策树分类报告：")
print(classification_report(y_tree_test, y_tree_pred, zero_division=1))


# ---------------------- 5. 结果分析 ----------------------
# 输出决策树的特征重要性（每个特征对预测的贡献程度，总和为1）
print("\n决策树特征重要性：")
for name, imp in zip(X_tree_train.columns, tree.feature_importances_):
    print(f"{name}：{imp:.4f}")  # 数值越大，该特征对信用等级预测的影响越大

# 可视化决策树（仅显示前两层，避免图形过于复杂）
plt.figure(figsize=(12, 8))  # 设置画布大小
# 绘制决策树：特征名称、类别名称、填充颜色、最大深度2层、字体大小10
plot_tree(tree, feature_names=X_tree_train.columns, class_names=["D", "C", "B", "A"], 
          filled=True, max_depth=2, fontsize=10)
plt.title("信用等级预测决策树（前两层）")  # 添加标题
plt.show()  # 显示图形
