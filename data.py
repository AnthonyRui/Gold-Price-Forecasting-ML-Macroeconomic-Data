import pandas as pd

# 1. 读取数据
df = pd.read_csv("cn_gdp.csv")

# 2. 设置列名（如果没有表头）
df.columns = ["Date", "Gold"]

# 3. 转换日期格式
df["Date"] = pd.to_datetime(df["Date"])

# 4. 按时间排序
df = df.sort_values("Date")

# 5. 去掉重复日期（保留第一条）
df = df.drop_duplicates(subset="Date", keep="first")

# 6. 设置 Date 为索引
df = df.set_index("Date")

# 7. 生成完整日期序列（按天）
full_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")

# 8. 重新索引，补全缺失日期
df = df.reindex(full_dates)

# 9. 用前值填充缺失
df["Gold"] = df["Gold"].ffill()

# 10. 重置索引
df = df.reset_index()

# 11. 重命名列名
df.rename(columns={"index": "Date"}, inplace=True)

# 12. 保存结果
df.to_csv("GDP.csv", index=False)

# 查看前几行
print(df.head())