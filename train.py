import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("total_featured.csv")
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# =========================
# 2. 清洗数据
# =========================
for col in df.columns:
    if col != "date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

# =========================
# 3. 再校验 Target
# Target = log(Gold_{t+1} / Gold_t)
# =========================
check_df = df[["date", "Gold", "Target"]].copy()
check_df["Gold_next"] = df["Gold"].shift(-1)
check_df["Target_check"] = np.log(check_df["Gold_next"] / check_df["Gold"])

target_ok = np.isclose(
    check_df["Target"].iloc[:-1],
    check_df["Target_check"].iloc[:-1],
    equal_nan=True
).all()

print("\n===== Target 再校验 =====")
print("Target 是否等于下一天对数收益率（前N-1行）:", target_ok)

if not target_ok:
    raise ValueError("Target 与下一天对数收益率不一致，请先重新运行 feature.py。")

print("\n前5行检查：")
print(df[["date", "Gold", "Target"]].head())

print("\n后5行检查：")
print(df[["date", "Gold", "Target"]].tail())

# =========================
# 4. 按时间切分
# =========================
train_df = df[(df["date"] >= "2001-09-29") & (df["date"] <= "2022-12-31")].copy()
val_df   = df[(df["date"] >= "2023-01-01") & (df["date"] <= "2023-12-31")].copy()
test_df  = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2025-12-29")].copy()

print("\n===== 数据集划分 =====")
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError("训练集/验证集/测试集有为空的情况，请检查日期范围。")

# =========================
# 5. 准备特征和标签
# =========================
feature_cols = [col for col in df.columns if col not in ["date", "Target"]]

X_train = train_df[feature_cols]
y_train = train_df["Target"]   # 对数收益率

X_val = val_df[feature_cols]
y_val = val_df["Target"]

X_test = test_df[feature_cols]
y_test = test_df["Target"]

print("\nNumber of features:", len(feature_cols))
print("Feature columns:")
print(feature_cols)

object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
if object_cols:
    raise ValueError(f"以下特征列仍为 object 类型: {object_cols}")

# =========================
# 6. 评估函数（价格层面）
# =========================
def evaluate_price_model(model_name, y_true_price, y_pred_price, dataset_name=""):
    mae = mean_absolute_error(y_true_price, y_pred_price)
    mse = mean_squared_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_price, y_pred_price)

    print(f"\n[{model_name}] {dataset_name} Results")
    print(f"MAE  = {mae:.6f}")
    print(f"MSE  = {mse:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R2   = {r2:.6f}")

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# =========================
# 7. 定义模型
# =========================
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="reg:squarederror"
)

rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

lgb_model = LGBMRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

# =========================
# 8. 训练模型
# =========================
print("\n===== 开始训练 XGBoost =====")
xgb_model.fit(X_train, y_train)

print("\n===== 开始训练 Random Forest =====")
rf_model.fit(X_train, y_train)

print("\n===== 开始训练 LightGBM =====")
lgb_model.fit(X_train, y_train)

# =========================
# 9. 预测“对数收益率”
# =========================
xgb_val_ret = xgb_model.predict(X_val)
rf_val_ret  = rf_model.predict(X_val)
lgb_val_ret = lgb_model.predict(X_val)

xgb_test_ret = xgb_model.predict(X_test)
rf_test_ret  = rf_model.predict(X_test)
lgb_test_ret = lgb_model.predict(X_test)

# =========================
# 10. 还原成“下一天价格”
# True_Next_Price = Gold_t * exp(Target)
# Pred_Next_Price = Gold_t * exp(Predicted_Log_Return)
# =========================
val_true_price = val_df["Gold"] * np.exp(y_val)
test_true_price = test_df["Gold"] * np.exp(y_test)

xgb_val_price = val_df["Gold"] * np.exp(xgb_val_ret)
rf_val_price  = val_df["Gold"] * np.exp(rf_val_ret)
lgb_val_price = val_df["Gold"] * np.exp(lgb_val_ret)

xgb_test_price = test_df["Gold"] * np.exp(xgb_test_ret)
rf_test_price  = test_df["Gold"] * np.exp(rf_test_ret)
lgb_test_price = test_df["Gold"] * np.exp(lgb_test_ret)

# =========================
# 11. 验证集结果（价格层面）
# =========================
xgb_val_result = evaluate_price_model("XGBoost", val_true_price, xgb_val_price, "Validation")
rf_val_result  = evaluate_price_model("RandomForest", val_true_price, rf_val_price, "Validation")
lgb_val_result = evaluate_price_model("LightGBM", val_true_price, lgb_val_price, "Validation")

val_results_df = pd.DataFrame([
    xgb_val_result,
    rf_val_result,
    lgb_val_result
])

val_results_df.to_csv("validation_results.csv", index=False, encoding="utf-8-sig")

print("\n验证集结果已保存到 validation_results.csv")
print(val_results_df)

# =========================
# 12. 测试集单模型结果（价格层面）
# =========================
xgb_test_result = evaluate_price_model("XGBoost", test_true_price, xgb_test_price, "Test")
rf_test_result  = evaluate_price_model("RandomForest", test_true_price, rf_test_price, "Test")
lgb_test_result = evaluate_price_model("LightGBM", test_true_price, lgb_test_price, "Test")

single_model_results_df = pd.DataFrame([
    xgb_test_result,
    rf_test_result,
    lgb_test_result
])

single_model_results_df.to_csv("single_model_test_results.csv", index=False, encoding="utf-8-sig")

print("\n单模型测试集结果已保存到 single_model_test_results.csv")
print(single_model_results_df)

# =========================
# 13. AEWE 权重（基于验证集“对数收益率”的 MAE）
# =========================
mae_xgb = mean_absolute_error(y_val, xgb_val_ret)
mae_rf  = mean_absolute_error(y_val, rf_val_ret)
mae_lgb = mean_absolute_error(y_val, lgb_val_ret)

print("\n===== Validation MAE (Log Return) =====")
print(f"XGBoost MAE       = {mae_xgb:.6f}")
print(f"Random Forest MAE = {mae_rf:.6f}")
print(f"LightGBM MAE      = {mae_lgb:.6f}")

eps = 1e-8

w_xgb = 1 / (mae_xgb + eps)
w_rf  = 1 / (mae_rf + eps)
w_lgb = 1 / (mae_lgb + eps)

weight_sum = w_xgb + w_rf + w_lgb
w_xgb, w_rf, w_lgb = w_xgb / weight_sum, w_rf / weight_sum, w_lgb / weight_sum

print("\n===== AEWE Weights =====")
print(f"XGBoost Weight       = {w_xgb:.6f}")
print(f"Random Forest Weight = {w_rf:.6f}")
print(f"LightGBM Weight      = {w_lgb:.6f}")
print(f"Weight Sum           = {w_xgb + w_rf + w_lgb:.6f}")

weights_df = pd.DataFrame({
    "Model": ["XGBoost", "RandomForest", "LightGBM"],
    "Validation_MAE_LogReturn": [mae_xgb, mae_rf, mae_lgb],
    "AEWE_Weight": [w_xgb, w_rf, w_lgb]
})

weights_df.to_csv("aewe_weights.csv", index=False, encoding="utf-8-sig")

print("\nAEWE 权重已保存到 aewe_weights.csv")
print(weights_df)

# =========================
# 14. AEWE 融合预测（先融合收益率，再还原价格）
# =========================
aewe_test_ret = (
    w_xgb * xgb_test_ret +
    w_rf  * rf_test_ret +
    w_lgb * lgb_test_ret
)

aewe_test_price = test_df["Gold"] * np.exp(aewe_test_ret)

aewe_test_result = evaluate_price_model("AEWE", test_true_price, aewe_test_price, "Test")

# =========================
# 15. 保存测试集预测结果
# =========================
result_df = test_df[["date", "Gold"]].copy()
result_df["True_Next_Price"] = test_true_price
result_df["XGBoost_Pred"] = xgb_test_price
result_df["RF_Pred"] = rf_test_price
result_df["LGBM_Pred"] = lgb_test_price
result_df["AEWE_Pred"] = aewe_test_price

result_df.to_csv("aewe_test_predictions_price.csv", index=False, encoding="utf-8-sig")

print("\n测试集价格预测已保存到 aewe_test_predictions_price.csv")
print(result_df.head())

# =========================
# 16. 保存最终结果汇总
# =========================
final_results_df = pd.DataFrame([
    xgb_test_result,
    rf_test_result,
    lgb_test_result,
    aewe_test_result
])

final_results_df.to_csv("model_results_summary_with_aewe.csv", index=False, encoding="utf-8-sig")

print("\n===== Final Summary =====")
print(final_results_df)