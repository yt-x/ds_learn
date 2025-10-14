import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 中文支持
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    使用 IQR 方法检测单列异常值

    参数:
    - data: Series 或 DataFrame 列
    - column: 列名（如果 data 是 DataFrame）
    - multiplier: IQR 乘数，默认 1.5

    返回:
    - outliers: 异常值 Series
    - bounds: 上下界字典
    - summary: 统计摘要
    """
    # 提取数据
    if isinstance(data, pd.DataFrame):
        series = data[column]
    else:
        series = data

    # 计算四分位数
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # 计算边界
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # 识别异常值
    outliers = series[(series < lower_bound) | (series > upper_bound)]

    # 统计信息
    summary = {
        "count": len(series),
        "outlier_count": len(outliers),
        "outlier_percentage": len(outliers) / len(series) * 100,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }

    bounds = {"lower": lower_bound, "upper": upper_bound}

    return outliers, bounds, summary


def visualize_outliers(df, column, results=None, figsize=(12, 8)):
    """
    可视化异常值检测结果
    """
    if results is None:
        outliers, bounds, summary = detect_outliers_iqr(df, column)
    else:
        outliers = results["outliers"]
        bounds = results["bounds"]
        summary = results["summary"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. 箱线图
    df.boxplot(column=column, ax=axes[0, 0])
    axes[0, 0].set_title(f"{column} - 箱线图")

    # 2. 直方图 + 边界线
    axes[0, 1].hist(df[column], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(
        bounds["lower"],
        color="red",
        linestyle="--",
        label=f"下界: {bounds['lower']:.2f}",
    )
    axes[0, 1].axvline(
        bounds["upper"],
        color="red",
        linestyle="--",
        label=f"上界: {bounds['upper']:.2f}",
    )
    axes[0, 1].set_title(f"{column} - 直方图与边界")
    axes[0, 1].legend()

    # 3. 散点图（显示异常值）
    normal_data = df[~df[column].isin(outliers)][column]
    outlier_data = outliers

    axes[1, 0].scatter(range(len(normal_data)), normal_data, alpha=0.6, label="正常值")
    axes[1, 0].scatter(
        range(len(normal_data), len(normal_data) + len(outlier_data)),
        outlier_data,
        color="red",
        label="异常值",
    )
    axes[1, 0].axhline(bounds["upper"], color="red", linestyle="--", alpha=0.5)
    axes[1, 0].set_title(f"{column} - 异常值散点图")
    axes[1, 0].legend()

    # 4. 统计信息
    axes[1, 1].axis("off")
    info_text = f"""统计摘要:
数据总数: {summary["count"]}
异常值数: {summary["outlier_count"]}
异常值比例: {summary["outlier_percentage"]:.2f}%
Q1: {summary["Q1"]:.2f}
Q3: {summary["Q3"]:.2f}
IQR: {summary["IQR"]:.2f}
下界: {summary["lower_bound"]:.2f}
上界: {summary["upper_bound"]:.2f}
"""
    axes[1, 1].text(
        0.1,
        0.9,
        info_text,
        transform=axes[1, 1].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


# 创建示例数据
np.random.seed(42)
data = pd.DataFrame(
    {
        "normal_data": np.random.normal(100, 15, 1000),
        "skewed_data": np.concatenate(
            [
                np.random.exponential(50, 950),  # 正常数据
                np.random.exponential(200, 50),  # 异常值
            ]
        ),
        "sales": np.concatenate(
            [
                np.random.normal(50000, 10000, 980),  # 正常销售
                [200000, 250000, 300000, 15000, 8000] * 4,  # 异常销售
            ]
        ),
    }
)

print("示例数据统计:")
print(data.describe())

# 检测 sales 列的异常值
outliers, bounds, summary = detect_outliers_iqr(data, "sales")

print("=== Sales 列异常值检测结果 ===")
print(f"数据总数: {summary['count']}")
print(f"异常值数量: {summary['outlier_count']}")
print(f"异常值比例: {summary['outlier_percentage']:.2f}%")
print(f"Q1: {summary['Q1']:.2f}")
print(f"Q3: {summary['Q3']:.2f}")
print(f"IQR: {summary['IQR']:.2f}")
print(f"下界: {summary['lower_bound']:.2f}")
print(f"上界: {summary['upper_bound']:.2f}")
print("\n异常值:")
print(outliers)


# 可视化 sales 列的异常值
visualize_outliers(data, "sales")
