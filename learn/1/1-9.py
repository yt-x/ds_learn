import pandas as pd
import numpy as np

# 1. 创建DataFrame：核燃料元件生产数据
data = {
    '批次号': ['FN2023001', 'FN2023002', 'FN2023003', 'FN2023004', 'FN2023005', 
              'FN2023006', 'FN2023007', 'FN2023008', 'FN2023009', 'FN2023010'],
    '燃料类型': ['UO2', 'MOX', 'UO2', 'UO2', 'MOX', 'UO2', 'MOX', 'UO2', 'MOX', 'UO2'],
    '生产车间': ['一车间', '二车间', '一车间', '三车间', '二车间', '三车间', '二车间', '一车间', '三车间', '一车间'],
    '生产数量': [500, 300, 450, 520, 280, 480, 320, 510, 290, 490],
    '合格率': [0.98, 0.96, 0.99, 0.97, 0.95, 0.98, 0.97, 0.99, 0.96, 0.98],
    '生产周期(天)': [15, 20, 14, 16, 21, 15, 19, 14, 22, 15],
    '生产成本(万元)': [1250, 1800, 1125, 1300, 1680, 1200, 1920, 1275, 1740, 1225]
}

df = pd.DataFrame(data)
print("1. 原始生产数据：")
print(df.head(), "\n")


# 2. 数据查询与筛选
# 2.1 筛选合格率98%及以上的批次
high_quality = df[df['合格率'] >= 0.98]
print("2.1 合格率≥98%的批次：")
print(high_quality[['批次号', '燃料类型', '合格率']], "\n")

# 2.2 筛选UO2类型且生产数量超过480的批次
uo2_large = df[(df['燃料类型'] == 'UO2') & (df['生产数量'] > 480)]
print("2.2 UO2类型且产量>480的批次：")
print(uo2_large[['批次号', '生产数量', '生产车间']], "\n")


# 3. 数据修改与新增列
# 3.1 计算合格数量和单位成本（确保无NaN后再转换为int）
df['合格数量'] = (df['生产数量'] * df['合格率']).astype(int)
df['单位成本(元/个)'] = (df['生产成本(万元)'] * 10000 / df['生产数量']).round(2)

# 3.2 修改生产周期异常值（超过20天的修正为20）
df.loc[df['生产周期(天)'] > 20, '生产周期(天)'] = 20
print("3. 修改后的数据（新增列和修正后）：")
print(df[['批次号', '合格数量', '单位成本(元/个)', '生产周期(天)']].head(), "\n")


# 4. 数据排序
# 按合格率降序、生产数量升序排序
sorted_df = df.sort_values(by=['合格率', '生产数量'], ascending=[False, True])
print("4. 按合格率降序、生产数量升序排序：")
print(sorted_df[['批次号', '燃料类型', '合格率', '生产数量']].head(), "\n")


# 5. 数据分组与聚合
# 5.1 按燃料类型分组，计算各指标平均值
fuel_type_stats = df.groupby('燃料类型').agg({
    '生产数量': 'mean',
    '合格率': 'mean',
    '生产成本(万元)': 'sum',
    '合格数量': 'sum'
}).round(2)
print("5.1 按燃料类型统计：")
print(fuel_type_stats, "\n")

# 5.2 按车间和燃料类型分组，计算平均生产周期
workshop_fuel_stats = df.groupby(['生产车间', '燃料类型'])['生产周期(天)'].mean().unstack()
print("5.2 车间×燃料类型的平均生产周期：")
print(workshop_fuel_stats.fillna('-'), "\n")


# 6. 透视表分析（修复NaN转换问题）
pivot_table = df.pivot_table(
    index='生产车间',
    columns='燃料类型',
    values='合格数量',
    aggfunc='sum',
    margins=True,
    margins_name='合计'
)

# 处理透视表中的NaN值后再转换为整数
pivot_table_clean = pivot_table.fillna(0).astype(int)  # 关键修复：先用0填充NaN

print("6. 各车间不同燃料类型的合格数量汇总：")
print(pivot_table_clean)

# 7. 数据删除

# 7.1 删除列（删除"合格数量"列）
df = df.drop(columns=["合格数量"])
# 7.2 删除行（删除生产成本高于1900万元的数据）
df = df.drop(df[df["生产成本(万元)"] >=1900].index)
print("7.删除后的数据：")
print(df)
print(df[["批次号","燃料类型", "生产成本(万元)"]])

