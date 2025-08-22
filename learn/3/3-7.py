#例3-7
import seaborn as sns
import pandas as pd
# 读入seaborn自身带的数据
try:
    # 尝试直接加载数据集
    fmri = sns.load_dataset("fmri")
except Exception as e:
    print(f"加载数据集失败: {e}")
    print("尝试使用手动构造的模拟数据...")
    #fmri =pd.read_csv("data/fmri.csv")
    # 手动构造模拟数据（当无法下载时使用）
sns.relplot(x="timepoint", y="signal", kind="line",data=fmri)
