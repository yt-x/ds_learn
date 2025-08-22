
#例3-2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_style({"font.sans-serif": "SimHei"})
data = [    ["2022-01-01", 16, 5],    ["2022-01-02", 15, 7],
    ["2022-01-03", 16, 8],    ["2022-01-04", 18, 6],
    ["2022-01-05", 17, 8]] # 气温数据（单位：度）
df = pd.DataFrame(data, columns=["日期", "广州", "北京"])
sns.lineplot(x=df["日期"], y=df["广州"])
sns.lineplot(x=df["日期"], y=df["北京"])
plt.show()

