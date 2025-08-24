import os
from pathlib import Path

import pandas as pd


def analyze_assets(file_path):
    """处理资产设备表并进行统计分析"""
    # 1. 读取Excel文件
    try:
        # 根据文件后缀选择合适的引擎
        file_ext = Path(file_path).suffix.lower()
        engine = 'xlrd' if file_ext == '.xls' else 'openpyxl'
        df = pd.read_excel(file_path, engine=engine)
        print(f"成功读取文件: {file_path}, 原始数据共 {len(df)} 条记录, {len(df.columns)} 列")
        print("原始数据前5行:")
        print(df.head())
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        return None
    # 2. 数据清洗
    print("\n" + "=" * 50 + "\n数据清洗:")
    # 2.1 处理缺失值
    missing_values = df.isnull().sum()
    print("\n各列缺失值数量:")
    print(missing_values[missing_values > 0])  # 只显示有缺失值的列
    if missing_values.sum() > 0:
        # 对于数值列(数量、价值), 用均值填充
        numeric_cols = ['数量', '价值']
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                #df[col] = df[col].fillna(df[col].mean())
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"已用均值填充'{col}'列的缺失值")
        print(df)
        # 对于日期列, 用众数填充
        date_cols = ['财务入账日期']
        for col in date_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"已用众数填充'{col}'列的缺失值")
        # 对于其他列, 用'前值'填充
        other_cols = [col for col in df.columns if col not in numeric_cols + date_cols]
        for col in other_cols:
            if df[col].isnull().sum() > 0:
                # df[col] = df[col].fillna('未知')
                df[col] = df[col].bfill()
                print(f"已用'前值'填充'{col}'列的缺失值")
    
    # 2.2 处理重复值
    duplicate_rows = df.duplicated().sum()
    print(f"\n重复记录数量: {duplicate_rows}")
    if duplicate_rows > 0:
        df.drop_duplicates(inplace=True)
        print(f"已删除所有重复记录, 剩余 {len(df)} 条记录")
   
    # 2.3 数据类型转换
    try:
        df['财务入账日期'] = pd.to_datetime(df['财务入账日期'])
        df['入账年份'] = df['财务入账日期'].dt.year
        df['入账月份'] = df['财务入账日期'].dt.month
        print("\n数据类型转换完成, 已添加'入账年份'和'入账月份'列")
    except Exception as e:
        print(f"日期转换警告: {str(e)}")

    # 3. 统计分析
    print("\n" + "=" * 50 + "\n统计分析结果:")
    
    # 3.1 基本统计量
    print("\n数值型数据基本统计量:")
    print(df[['数量', '价值']].describe().round(2))
    
    # 3.2 按资产分类统计
    print("\n按资产分类统计:")
    category_stats = df.groupby('资产分类').agg({
        '资产编号': 'count',  # 设备数量
        '数量': 'sum',  # 总数量
        '价值': ['sum', 'mean']  # 总价值和平均价值
    }).round(2)
    category_stats.columns = ['设备种类数', '总数量', '总价值', '平均价值']
    print(category_stats)
    # 3.3 按使用部门统计
    print("\n按使用部门统计:")
    dept_stats = df.groupby('使用部门').agg({
        '资产编号': 'count',  # 设备数量
        '价值': 'sum'  # 总价值
    }).round(2)
    dept_stats.columns = ['设备数量', '总价值(元)']
    # 计算各部门资产占比
    dept_stats['价值占比(%)'] = (dept_stats['总价值(元)'] /
                                 dept_stats['总价值(元)'].sum() * 100).round(2)
    print(dept_stats)
    # 3.4 按年份统计
    if '入账年份' in df.columns:
        print("\n按入账年份统计:")
        year_stats = df.groupby('入账年份').agg({
            '资产编号': 'count',  # 当年新增设备数
            '价值': 'sum'  # 当年新增资产总价值
        }).round(2)
        year_stats.columns = ['新增设备数', '新增资产价值(元)']
        print(year_stats)

    # 4. 保存清洗后的数据
    output_file = r"清洗后的资产设备表.xlsx"
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\n清洗后的数据集已保存至: {output_file}")
    return df
if __name__ == "__main__":
    # 资产设备表路径
    input_file = r"资产设备表.xls"
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 '{input_file}' 不存在！")
    else:
        # 执行分析
        analyze_assets(input_file)
