import pandas as pd
import numpy as np

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("total.csv")
df.columns = df.columns.str.strip()

# =========================
# 2. 日期处理
# =========================
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# =========================
# 3. 检查基础列
# =========================
required_cols = ["Gold", "USD", "Yield", "EFFR", "CPI", "PCE", "GDP"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"缺少以下必要列: {missing_cols}")

# =========================
# 4. 原始变量统一转数值
# =========================
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 5. 构造 Lag 特征
# =========================
df["Gold_lag1"] = df["Gold"].shift(1)
df["Gold_lag3"] = df["Gold"].shift(3)
df["Gold_lag5"] = df["Gold"].shift(5)
df["Gold_lag10"] = df["Gold"].shift(10)
df["Gold_lag20"] = df["Gold"].shift(20)
df["Gold_lag60"] = df["Gold"].shift(60)

df["USD_lag1"] = df["USD"].shift(1)
df["USD_lag3"] = df["USD"].shift(3)
df["USD_lag5"] = df["USD"].shift(5)

df["Yield_lag1"] = df["Yield"].shift(1)
df["Yield_lag3"] = df["Yield"].shift(3)
df["Yield_lag5"] = df["Yield"].shift(5)

df["EFFR_lag1"] = df["EFFR"].shift(1)
df["EFFR_lag3"] = df["EFFR"].shift(3)

df["CPI_lag1"] = df["CPI"].shift(1)
df["PCE_lag1"] = df["PCE"].shift(1)
df["GDP_lag1"] = df["GDP"].shift(1)

# =========================
# 6. 构造 Rolling 特征
# =========================
df["Gold_ma5"] = df["Gold"].rolling(window=5).mean()
df["Gold_ma10"] = df["Gold"].rolling(window=10).mean()
df["Gold_ma20"] = df["Gold"].rolling(window=20).mean()
df["Gold_ma60"] = df["Gold"].rolling(window=60).mean()
df["Gold_ma120"] = df["Gold"].rolling(window=120).mean()

df["Gold_std5"] = df["Gold"].rolling(window=5).std()
df["Gold_std10"] = df["Gold"].rolling(window=10).std()
df["Gold_std20"] = df["Gold"].rolling(window=20).std()

df["USD_ma5"] = df["USD"].rolling(window=5).mean()
df["Yield_ma5"] = df["Yield"].rolling(window=5).mean()

# =========================
# 7. 构造变化率 / 差分特征
# =========================
df["Gold_return_1d"] = df["Gold"].pct_change(periods=1)
df["Gold_return_5d"] = df["Gold"].pct_change(periods=5)
df["Gold_return_20d"] = df["Gold"].pct_change(periods=20)

df["USD_return_1d"] = df["USD"].pct_change(periods=1)
df["USD_return_5d"] = df["USD"].pct_change(periods=5)

df["Yield_change_1d"] = df["Yield"].diff(periods=1)
df["Yield_change_5d"] = df["Yield"].diff(periods=5)
df["EFFR_change_1d"] = df["EFFR"].diff(periods=1)

df["CPI_change_1d"] = df["CPI"].diff(periods=1)
df["PCE_change_1d"] = df["PCE"].diff(periods=1)
df["GDP_change_1d"] = df["GDP"].diff(periods=1)

df["CPI_return_1d"] = df["CPI"].pct_change(periods=1)
df["PCE_return_1d"] = df["PCE"].pct_change(periods=1)
df["GDP_return_1d"] = df["GDP"].pct_change(periods=1)

# =========================
# 8. 趋势 / 动量特征
# =========================
df["Gold_mom20"] = df["Gold"] / df["Gold"].shift(20) - 1
df["Gold_mom60"] = df["Gold"] / df["Gold"].shift(60) - 1

df["Gold_to_ma20"] = df["Gold"] / df["Gold_ma20"] - 1
df["Gold_to_ma60"] = df["Gold"] / df["Gold_ma60"] - 1
df["Gold_to_ma120"] = df["Gold"] / df["Gold_ma120"] - 1

# =========================
# 9. 时间特征
# =========================
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["day_of_week"] = df["date"].dt.dayofweek

# =========================
# 10. 全部非日期列统一转数值
# =========================
for col in df.columns:
    if col != "date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 替换 inf / -inf
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# 11. 先删除特征工程产生的缺失值
# =========================
df = df.dropna().reset_index(drop=True)

# =========================
# 12. 最后生成 Target（下一天对数收益率）
# Target = log(Gold_{t+1} / Gold_t)
# =========================
df["Target"] = np.log(df["Gold"].shift(-1) / df["Gold"])

# 删除最后一行（因为最后一天没有下一天）
df = df.dropna().reset_index(drop=True)

# =========================
# 13. 校验 Target 是否正确
# =========================
check_df = df[["date", "Gold", "Target"]].copy()
check_df["Gold_next"] = df["Gold"].shift(-1)
check_df["Target_check"] = np.log(check_df["Gold_next"] / check_df["Gold"])

target_ok = np.isclose(
    check_df["Target"].iloc[:-1],
    check_df["Target_check"].iloc[:-1],
    equal_nan=True
).all()

print("\n===== Target 校验（前10行） =====")
print(check_df.head(10))

print("\n===== Target 校验（后10行） =====")
print(check_df.tail(10))

print("\nTarget 是否等于下一天对数收益率（前N-1行）:", target_ok)

# =========================
# 14. 保存
# =========================
df.to_csv("total_featured.csv", index=False, encoding="utf-8-sig")

print("\n特征工程完成。")
print("处理后数据形状:", df.shape)

print("\n前5行：")
print(df.head())

print("\n生成的列名：")
print(df.columns.tolist())