import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = {
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'device': ['Mobile', 'Desktop', 'Tablet', 'Mobile', 'Desktop', 'Tablet', 'Mobile', 'Desktop', 'Tablet', 'Mobile']
}

df = pd.DataFrame(data)

# 2 独热编码
# 初始化 OneHotEncoder
encoder = OneHotEncoder()
# 对 gender 进行独热编码
gender_encoded = encoder.fit_transform(df[['gender']]).toarray()
gender_encoded_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['gender']))

# 对 device 进行独热编码
device_encoded = encoder.fit_transform(df[['device']]).toarray()
device_encoded_df = pd.DataFrame(device_encoded, columns=encoder.get_feature_names_out(['device']))
# 将独热编码后的数据合并到原数据框
df_encoded = pd.concat([df, gender_encoded_df, device_encoded_df], axis=1)

# 3. 绘制分类柱状图
fig = plt.figure(figsize=(12, 5), dpi=300)

# 用户性别分布
plt.subplot(1, 2, 1)
gender_plot = sns.countplot(x=df_encoded[['gender_Female', 'gender_Male']].idxmax(axis=1))
plt.title("用户性别分布", fontsize=14, fontweight='bold')
plt.xlabel('性别', fontsize=12)
plt.ylabel('数量', fontsize=12)
# 设置x轴标签
gender_plot.set_xticklabels(['Female', 'Male'])

# 用户设备分布
plt.subplot(1, 2, 2)
device_plot = sns.countplot(x=df_encoded[['device_Mobile', 'device_Desktop', 'device_Tablet']].idxmax(axis=1))
plt.title("用户设备分布", fontsize=14, fontweight='bold')
plt.xlabel('设备类型', fontsize=12)
plt.ylabel('数量', fontsize=12)
# 设置x轴标签
device_plot.set_xticklabels(['Mobile', 'Desktop', 'Tablet'])

# 调整子图间距
plt.tight_layout(pad=3.0)

# 添加整体标题
fig.suptitle('用户特征分布分析', fontsize=16, fontweight='bold', y=1.02)

# 显示图形
plt.show()