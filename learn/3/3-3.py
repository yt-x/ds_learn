#例3-3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns# 设置
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})# 气温数据（单位：度）
data = [    ["2022-01-01", 16, 5],    ["2022-01-02", 15, 7],
    ["2022-01-03", 16, 8],    ["2022-01-04", 18, 6],
    ["2022-01-05", 17, 8]]
df = pd.DataFrame(data, columns=["日期", "广州", "北京"])
df.set_index("日期", inplace=True)
sns.lineplot(data=df)
plt.show()

