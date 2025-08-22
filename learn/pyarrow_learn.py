import pyarrow as pa
import pyarrow.parquet as pq

# 创建一个 Table
data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
table = pa.Table.from_pydict(data)

# 将表写入 Parquet 文件
pq.write_table(table, "example.parquet")

# 从 Parquet 文件中读取表
table_read = pq.read_table("example.parquet")
print(table_read)
