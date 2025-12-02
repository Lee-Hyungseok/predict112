"""
하이퍼파라미터 튜닝으로 최고 성능 모델 만들기
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("하이퍼파라미터 튜닝으로 최고 성능 모델 만들기")
print("="*80)

# 데이터 로드
df = pd.read_csv('reg_predict112.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# 2022년 9월 데이터 분리
predict_data = df[(df['year'] == 2022) & (df['month'] == 9)].copy()
train_data = df[~((df['year'] == 2022) & (df['month'] == 9))].copy()

target_cols = ['seogang', 'gongdeok', 'worldcup', 'hongik', 'yonggang']
feature_cols = [col for col in df.columns if col not in ['date', 'year', 'month'] + target_cols]

# 결과 저장
tuned_results = []
final_predictions = {}

for target in target_cols:
    print(f"\n{'='*80}")
    print(f"타겟: {target}")
    print(f"{'='*80}")

    X_train = train_data[feature_cols]
    y_train = train_data[target]
    X_pred = predict_data[feature_cols]

    # 학습/검증 분리
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    X_val_split = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val_split = y_train[split_idx:]

    best_models = {}

    # 1. CatBoost 튜닝 (대부분의 역에서 최고 성능)
    print("\n[1] CatBoost 하이퍼파라미터 튜닝...")
    cat_params = {
        'iterations': [300, 500],
        'depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
    }

    cat_model = cb.CatBoostRegressor(random_state=42, verbose=0)
    cat_grid = RandomizedSearchCV(cat_model, cat_params, n_iter=10, cv=3,
                                  scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
    cat_grid.fit(X_train_split, y_train_split)

    best_cat = cat_grid.best_estimator_
    cat_pred = best_cat.predict(X_val_split)
    cat_mae = mean_absolute_error(y_val_split, cat_pred)
    cat_rmse = np.sqrt(mean_squared_error(y_val_split, cat_pred))
    cat_r2 = r2_score(y_val_split, cat_pred)
    cat_accuracy = 100 * (1 - cat_mae / y_val_split.mean())

    print(f"  Best params: {cat_grid.best_params_}")
    print(f"  MAE: {cat_mae:.4f}, RMSE: {cat_rmse:.4f}, R²: {cat_r2:.4f}, 정확도: {cat_accuracy:.2f}%")

    best_cat.fit(X_train, y_train)
    cat_final_pred = best_cat.predict(X_pred)
    best_models['CatBoost'] = {
        'model': best_cat,
        'predictions': cat_final_pred,
        'mae': cat_mae,
        'rmse': cat_rmse,
        'r2': cat_r2,
        'accuracy': cat_accuracy
    }

    # 2. LightGBM 튜닝
    print("\n[2] LightGBM 하이퍼파라미터 튜닝...")
    lgb_params = {
        'n_estimators': [300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgb_grid = RandomizedSearchCV(lgb_model, lgb_params, n_iter=10, cv=3,
                                  scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
    lgb_grid.fit(X_train_split, y_train_split)

    best_lgb = lgb_grid.best_estimator_
    lgb_pred = best_lgb.predict(X_val_split)
    lgb_mae = mean_absolute_error(y_val_split, lgb_pred)
    lgb_rmse = np.sqrt(mean_squared_error(y_val_split, lgb_pred))
    lgb_r2 = r2_score(y_val_split, lgb_pred)
    lgb_accuracy = 100 * (1 - lgb_mae / y_val_split.mean())

    print(f"  Best params: {lgb_grid.best_params_}")
    print(f"  MAE: {lgb_mae:.4f}, RMSE: {lgb_rmse:.4f}, R²: {lgb_r2:.4f}, 정확도: {lgb_accuracy:.2f}%")

    best_lgb.fit(X_train, y_train)
    lgb_final_pred = best_lgb.predict(X_pred)
    best_models['LightGBM'] = {
        'model': best_lgb,
        'predictions': lgb_final_pred,
        'mae': lgb_mae,
        'rmse': lgb_rmse,
        'r2': lgb_r2,
        'accuracy': lgb_accuracy
    }

    # 3. XGBoost 튜닝
    print("\n[3] XGBoost 하이퍼파라미터 튜닝...")
    xgb_params = {
        'n_estimators': [300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_grid = RandomizedSearchCV(xgb_model, xgb_params, n_iter=10, cv=3,
                                  scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
    xgb_grid.fit(X_train_split, y_train_split)

    best_xgb = xgb_grid.best_estimator_
    xgb_pred = best_xgb.predict(X_val_split)
    xgb_mae = mean_absolute_error(y_val_split, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_pred))
    xgb_r2 = r2_score(y_val_split, xgb_pred)
    xgb_accuracy = 100 * (1 - xgb_mae / y_val_split.mean())

    print(f"  Best params: {xgb_grid.best_params_}")
    print(f"  MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}, 정확도: {xgb_accuracy:.2f}%")

    best_xgb.fit(X_train, y_train)
    xgb_final_pred = best_xgb.predict(X_pred)
    best_models['XGBoost'] = {
        'model': best_xgb,
        'predictions': xgb_final_pred,
        'mae': xgb_mae,
        'rmse': xgb_rmse,
        'r2': xgb_r2,
        'accuracy': xgb_accuracy
    }

    # 4. Random Forest 튜닝
    print("\n[4] Random Forest 하이퍼파라미터 튜닝...")
    rf_params = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = RandomizedSearchCV(rf_model, rf_params, n_iter=8, cv=3,
                                 scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
    rf_grid.fit(X_train_split, y_train_split)

    best_rf = rf_grid.best_estimator_
    rf_pred = best_rf.predict(X_val_split)
    rf_mae = mean_absolute_error(y_val_split, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_val_split, rf_pred))
    rf_r2 = r2_score(y_val_split, rf_pred)
    rf_accuracy = 100 * (1 - rf_mae / y_val_split.mean())

    print(f"  Best params: {rf_grid.best_params_}")
    print(f"  MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}, 정확도: {rf_accuracy:.2f}%")

    best_rf.fit(X_train, y_train)
    rf_final_pred = best_rf.predict(X_pred)
    best_models['Random Forest'] = {
        'model': best_rf,
        'predictions': rf_final_pred,
        'mae': rf_mae,
        'rmse': rf_rmse,
        'r2': rf_r2,
        'accuracy': rf_accuracy
    }

    # 5. 가중 앙상블 (성능 기반 가중치)
    print("\n[5] 가중 앙상블...")
    weights = []
    predictions = []
    for model_name, info in best_models.items():
        weight = 1 / (info['mae'] + 1e-10)  # MAE가 낮을수록 높은 가중치
        weights.append(weight)
        predictions.append(info['predictions'])

    weights = np.array(weights) / np.sum(weights)
    weighted_pred_val = np.average([cat_pred, lgb_pred, xgb_pred, rf_pred], axis=0, weights=weights)
    weighted_pred_final = np.average(predictions, axis=0, weights=weights)

    ens_mae = mean_absolute_error(y_val_split, weighted_pred_val)
    ens_rmse = np.sqrt(mean_squared_error(y_val_split, weighted_pred_val))
    ens_r2 = r2_score(y_val_split, weighted_pred_val)
    ens_accuracy = 100 * (1 - ens_mae / y_val_split.mean())

    print(f"  가중치: {[f'{w:.3f}' for w in weights]}")
    print(f"  MAE: {ens_mae:.4f}, RMSE: {ens_rmse:.4f}, R²: {ens_r2:.4f}, 정확도: {ens_accuracy:.2f}%")

    best_models['Weighted Ensemble'] = {
        'predictions': weighted_pred_final,
        'mae': ens_mae,
        'rmse': ens_rmse,
        'r2': ens_r2,
        'accuracy': ens_accuracy
    }

    # 최고 모델 선택
    best_model_name = min(best_models.items(), key=lambda x: x[1]['mae'])[0]
    print(f"\n★★★ 최고 성능 모델: {best_model_name} (MAE: {best_models[best_model_name]['mae']:.4f}, 정확도: {best_models[best_model_name]['accuracy']:.2f}%)")

    # 결과 저장
    for model_name, info in best_models.items():
        tuned_results.append({
            'Station': target,
            'Model': model_name,
            'MAE': info['mae'],
            'RMSE': info['rmse'],
            'R2': info['r2'],
            'Accuracy': info['accuracy']
        })

    final_predictions[target] = {
        'best_model': best_model_name,
        'predictions': best_models[best_model_name]['predictions'],
        'all_predictions': {k: v['predictions'] for k, v in best_models.items()}
    }

# 결과 DataFrame
tuned_df = pd.DataFrame(tuned_results)

# 상세 결과 저장
with open('hyperparameter_tuning_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("하이퍼파라미터 튜닝 후 최종 결과\n")
    f.write("="*80 + "\n\n")

    f.write("[전체 튜닝된 모델 성능]\n\n")
    f.write(tuned_df.to_string(index=False))
    f.write("\n\n")

    f.write("[각 역별 최고 성능 모델]\n\n")
    for station in target_cols:
        station_data = tuned_df[tuned_df['Station'] == station]
        best = station_data.loc[station_data['MAE'].idxmin()]
        f.write(f"{station}:\n")
        f.write(f"  최고 모델: {best['Model']}\n")
        f.write(f"  MAE: {best['MAE']:.4f}\n")
        f.write(f"  RMSE: {best['RMSE']:.4f}\n")
        f.write(f"  R²: {best['R2']:.4f}\n")
        f.write(f"  예측 정확도: {best['Accuracy']:.2f}%\n\n")

    f.write("[2022년 9월 최종 예측 결과]\n\n")
    pred_df = predict_data[['date']].copy()
    for station in target_cols:
        pred_df[f'{station}_predicted'] = final_predictions[station]['predictions']

    f.write(pred_df.to_string(index=False))
    f.write("\n\n")

    f.write("[2022년 9월 평균 예측 신고량]\n\n")
    for station in target_cols:
        avg_pred = final_predictions[station]['predictions'].mean()
        f.write(f"{station}: {avg_pred:.2f}건 (모델: {final_predictions[station]['best_model']})\n")

print("\n" + "="*80)
print("하이퍼파라미터 튜닝 완료!")
print("="*80)
print("\n결과가 'hyperparameter_tuning_results.txt' 파일에 저장되었습니다.")

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Tuned Model Performance Comparison', fontsize=16, fontweight='bold')

for idx, station in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    station_data = tuned_df[tuned_df['Station'] == station].sort_values('MAE')

    colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(station_data))]
    bars = ax.barh(station_data['Model'], station_data['MAE'], color=colors)

    ax.set_xlabel('MAE', fontsize=10)
    ax.set_title(f'{station} Station', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    for bar, mae, acc in zip(bars, station_data['MAE'], station_data['Accuracy']):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {mae:.2f} ({acc:.1f}%)', va='center', fontsize=8)

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('tuned_model_comparison.png', dpi=300, bbox_inches='tight')
print("그래프 저장: tuned_model_comparison.png")

# 정확도 히트맵
fig, ax = plt.subplots(figsize=(12, 6))
import seaborn as sns
pivot_accuracy = tuned_df.pivot(index='Model', columns='Station', values='Accuracy')
sns.heatmap(pivot_accuracy, annot=True, fmt='.1f', cmap='RdYlGn', center=80,
            cbar_kws={'label': 'Accuracy %'}, ax=ax)
ax.set_title('Tuned Model Accuracy Heatmap (%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tuned_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
print("그래프 저장: tuned_accuracy_heatmap.png")

print("\n모든 작업 완료!")
