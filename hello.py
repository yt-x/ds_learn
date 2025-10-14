import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'experience': [2, 5, 8, 12, 15],
    'performance': ['A', 'B', 'A', 'C', 'B']
})

scaler = MinMaxScaler()

numeric_cols = df.select_dtypes(include=[np.number]).columns

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("MinMaxScaler 标准化后的数据:")
print(df)