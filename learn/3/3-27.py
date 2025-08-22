import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
sns.set(font="SimHei", font_scale=1.1)

# 准备数据
np.random.seed(42)  # 设置随机种子，保证结果可复现
data = {
    "生产车间": np.repeat(["一车间", "二车间", "三车间"], 100),  # 3个车间各100条数据
    "铀浓度(%)": np.concatenate([
        np.random.normal(4.8, 0.3, 100),  # 一车间：均值4.8，标准差0.3（稳定）
        np.random.normal(4.8, 0.5, 100),  # 二车间：标准差0.5（波动较大）
        np.concatenate([                  # 三车间：含5个高浓度异常值
            np.random.normal(4.8, 0.3, 95),
            np.random.normal(6.0, 0.2, 5)
        ])
    ]),
    "质量等级": np.random.choice(["优", "良", "合格"], 300, p=[0.2, 0.5, 0.3])  # 模拟质量等级分布
}
df = pd.DataFrame(data)  # 转换为DataFrame


# ----------------------
# 基础箱线图
# ----------------------
plt.figure(figsize=(10, 6))  # 创建画布，设置大小
sns.boxplot(
    data=df,
    x="生产车间",         # x轴：分类变量（生产车间）
    y="铀浓度(%)",        # y轴：数值变量（铀浓度）
    hue="生产车间",       # 关键修正：将x变量赋值给hue，与x保持一致
    palette="Set2",       # 颜色方案（现在合法使用）
    legend=False,         # 隐藏图例（避免与x轴标签重复）
    showfliers=True       # 显示异常值
)
plt.title("各车间燃料棒铀浓度分布", pad=15)  # 标题
plt.ylabel("铀浓度(%)")                     # y轴标签
plt.axhline(y=4.8, linestyle="--", color="red", alpha=0.7, label="标准浓度")  # 参考线
plt.legend()  # 显示参考线的图例
plt.show()    # 展示图表


# ----------------------
# 分组箱线图（本身使用hue的情况不受影响）
# ----------------------
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x="生产车间",
    y="铀浓度(%)",
    hue="质量等级",     # 按质量等级进一步分组（本身已有hue，无警告）
    palette="coolwarm", # 颜色方案（合法使用）
    showfliers=False    # 隐藏异常值，聚焦主体数据
)
plt.title("各车间不同质量等级的铀浓度分布", pad=15)
plt.ylabel("铀浓度(%)")
plt.legend(title="质量等级")  # 显示质量等级的图例
plt.show()
