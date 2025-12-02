"""
2022년 9월 신고량 예측을 위한 다양한 모델 비교 및 최적화
목적: 최고 예측 정확도를 가진 모델 찾기
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# XGBoost, LightGBM, CatBoost 임포트
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except:
    HAS_CB = False

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("2022년 9월 신고량 예측 모델 비교 실험")
print("="*80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드 및 전처리")
df = pd.read_csv('reg_predict112.csv')
print(f"데이터 shape: {df.shape}")

# 날짜 파싱
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# 2022년 9월 데이터 분리 (예측 대상)
target_date = '2022-09'
predict_data = df[(df['year'] == 2022) & (df['month'] == 9)].copy()
train_data = df[~((df['year'] == 2022) & (df['month'] == 9))].copy()

print(f"학습 데이터: {train_data.shape}")
print(f"예측 대상 데이터: {predict_data.shape}")

# 타겟 변수들 (5개 역의 신고량)
target_cols = ['seogang', 'gongdeok', 'worldcup', 'hongik', 'yonggang']

# 특성 변수들 (날짜, 타겟 제외)
feature_cols = [col for col in df.columns if col not in ['date', 'year', 'month'] + target_cols]
print(f"\n특성 변수 개수: {len(feature_cols)}")
print(f"특성 변수: {feature_cols}")

# 결과 저장용
results = []
all_predictions = {}

# 각 역에 대해 모델 학습
for target in target_cols:
    print(f"\n{'='*80}")
    print(f"타겟: {target}")
    print(f"{'='*80}")

    # 학습/검증 데이터 분리
    X_train = train_data[feature_cols]
    y_train = train_data[target]
    X_pred = predict_data[feature_cols]

    # 학습/검증 분리 (시계열이므로 최근 20% 검증용)
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    X_val_split = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val_split = y_train[split_idx:]

    print(f"\n학습 세트: {X_train_split.shape}, 검증 세트: {X_val_split.shape}")

    # 스케일링 (Neural Network용)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_pred_scaled = scaler.transform(X_pred)

    station_results = {}
    station_predictions = {}

    # 1. Random Forest
    print("\n[1] Random Forest")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                               min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train_split, y_train_split)
    rf_pred = rf.predict(X_val_split)
    rf_mae = mean_absolute_error(y_val_split, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_val_split, rf_pred))
    rf_r2 = r2_score(y_val_split, rf_pred)
    rf_accuracy = 100 * (1 - rf_mae / y_val_split.mean())

    print(f"  MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}, 정확도: {rf_accuracy:.2f}%")

    # 전체 데이터로 재학습 후 예측
    rf.fit(X_train, y_train)
    rf_final_pred = rf.predict(X_pred)
    station_predictions['Random Forest'] = rf_final_pred
    station_results['Random Forest'] = {
        'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2, 'Accuracy': rf_accuracy
    }

    # 2. Gradient Boosting
    print("\n[2] Gradient Boosting")
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                   min_samples_split=5, min_samples_leaf=2, random_state=42)
    gb.fit(X_train_split, y_train_split)
    gb_pred = gb.predict(X_val_split)
    gb_mae = mean_absolute_error(y_val_split, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_val_split, gb_pred))
    gb_r2 = r2_score(y_val_split, gb_pred)
    gb_accuracy = 100 * (1 - gb_mae / y_val_split.mean())

    print(f"  MAE: {gb_mae:.4f}, RMSE: {gb_rmse:.4f}, R²: {gb_r2:.4f}, 정확도: {gb_accuracy:.2f}%")

    gb.fit(X_train, y_train)
    gb_final_pred = gb.predict(X_pred)
    station_predictions['Gradient Boosting'] = gb_final_pred
    station_results['Gradient Boosting'] = {
        'MAE': gb_mae, 'RMSE': gb_rmse, 'R2': gb_r2, 'Accuracy': gb_accuracy
    }

    # 3. XGBoost
    if HAS_XGB:
        print("\n[3] XGBoost")
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb_model.fit(X_train_split, y_train_split)
        xgb_pred = xgb_model.predict(X_val_split)
        xgb_mae = mean_absolute_error(y_val_split, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_pred))
        xgb_r2 = r2_score(y_val_split, xgb_pred)
        xgb_accuracy = 100 * (1 - xgb_mae / y_val_split.mean())

        print(f"  MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}, 정확도: {xgb_accuracy:.2f}%")

        xgb_model.fit(X_train, y_train)
        xgb_final_pred = xgb_model.predict(X_pred)
        station_predictions['XGBoost'] = xgb_final_pred
        station_results['XGBoost'] = {
            'MAE': xgb_mae, 'RMSE': xgb_rmse, 'R2': xgb_r2, 'Accuracy': xgb_accuracy
        }

    # 4. LightGBM
    if HAS_LGB:
        print("\n[4] LightGBM")
        lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        lgb_model.fit(X_train_split, y_train_split)
        lgb_pred = lgb_model.predict(X_val_split)
        lgb_mae = mean_absolute_error(y_val_split, lgb_pred)
        lgb_rmse = np.sqrt(mean_squared_error(y_val_split, lgb_pred))
        lgb_r2 = r2_score(y_val_split, lgb_pred)
        lgb_accuracy = 100 * (1 - lgb_mae / y_val_split.mean())

        print(f"  MAE: {lgb_mae:.4f}, RMSE: {lgb_rmse:.4f}, R²: {lgb_r2:.4f}, 정확도: {lgb_accuracy:.2f}%")

        lgb_model.fit(X_train, y_train)
        lgb_final_pred = lgb_model.predict(X_pred)
        station_predictions['LightGBM'] = lgb_final_pred
        station_results['LightGBM'] = {
            'MAE': lgb_mae, 'RMSE': lgb_rmse, 'R2': lgb_r2, 'Accuracy': lgb_accuracy
        }

    # 5. CatBoost
    if HAS_CB:
        print("\n[5] CatBoost")
        cat_model = cb.CatBoostRegressor(iterations=200, depth=5, learning_rate=0.1,
                                         random_state=42, verbose=0)
        cat_model.fit(X_train_split, y_train_split)
        cat_pred = cat_model.predict(X_val_split)
        cat_mae = mean_absolute_error(y_val_split, cat_pred)
        cat_rmse = np.sqrt(mean_squared_error(y_val_split, cat_pred))
        cat_r2 = r2_score(y_val_split, cat_pred)
        cat_accuracy = 100 * (1 - cat_mae / y_val_split.mean())

        print(f"  MAE: {cat_mae:.4f}, RMSE: {cat_rmse:.4f}, R²: {cat_r2:.4f}, 정확도: {cat_accuracy:.2f}%")

        cat_model.fit(X_train, y_train)
        cat_final_pred = cat_model.predict(X_pred)
        station_predictions['CatBoost'] = cat_final_pred
        station_results['CatBoost'] = {
            'MAE': cat_mae, 'RMSE': cat_rmse, 'R2': cat_r2, 'Accuracy': cat_accuracy
        }

    # 6. Neural Network (MLP)
    print("\n[6] Neural Network (MLP)")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                       solver='adam', alpha=0.001, batch_size=32, learning_rate='adaptive',
                       max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_train_scaled, y_train_split)
    mlp_pred = mlp.predict(X_val_scaled)
    mlp_mae = mean_absolute_error(y_val_split, mlp_pred)
    mlp_rmse = np.sqrt(mean_squared_error(y_val_split, mlp_pred))
    mlp_r2 = r2_score(y_val_split, mlp_pred)
    mlp_accuracy = 100 * (1 - mlp_mae / y_val_split.mean())

    print(f"  MAE: {mlp_mae:.4f}, RMSE: {mlp_rmse:.4f}, R²: {mlp_r2:.4f}, 정확도: {mlp_accuracy:.2f}%")

    X_train_all_scaled = scaler.fit_transform(X_train)
    mlp.fit(X_train_all_scaled, y_train)
    mlp_final_pred = mlp.predict(X_pred_scaled)
    station_predictions['Neural Network'] = mlp_final_pred
    station_results['Neural Network'] = {
        'MAE': mlp_mae, 'RMSE': mlp_rmse, 'R2': mlp_r2, 'Accuracy': mlp_accuracy
    }

    # 7. 앙상블 (평균)
    print("\n[7] Ensemble (평균)")
    ensemble_val_pred = np.mean([rf_pred, gb_pred] +
                                 ([xgb_pred] if HAS_XGB else []) +
                                 ([lgb_pred] if HAS_LGB else []) +
                                 ([cat_pred] if HAS_CB else []) +
                                 [mlp_pred], axis=0)
    ens_mae = mean_absolute_error(y_val_split, ensemble_val_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_val_pred))
    ens_r2 = r2_score(y_val_split, ensemble_val_pred)
    ens_accuracy = 100 * (1 - ens_mae / y_val_split.mean())

    print(f"  MAE: {ens_mae:.4f}, RMSE: {ens_rmse:.4f}, R²: {ens_r2:.4f}, 정확도: {ens_accuracy:.2f}%")

    ensemble_final_pred = np.mean([rf_final_pred, gb_final_pred] +
                                   ([xgb_final_pred] if HAS_XGB else []) +
                                   ([lgb_final_pred] if HAS_LGB else []) +
                                   ([cat_final_pred] if HAS_CB else []) +
                                   [mlp_final_pred], axis=0)
    station_predictions['Ensemble'] = ensemble_final_pred
    station_results['Ensemble'] = {
        'MAE': ens_mae, 'RMSE': ens_rmse, 'R2': ens_r2, 'Accuracy': ens_accuracy
    }

    # 최고 성능 모델 찾기
    best_model = min(station_results.items(), key=lambda x: x[1]['MAE'])
    print(f"\n★ 최고 성능 모델: {best_model[0]} (MAE: {best_model[1]['MAE']:.4f}, 정확도: {best_model[1]['Accuracy']:.2f}%)")

    # 결과 저장
    for model_name, metrics in station_results.items():
        results.append({
            'Station': target,
            'Model': model_name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'Accuracy': metrics['Accuracy']
        })

    all_predictions[target] = {
        'predictions': station_predictions,
        'best_model': best_model[0]
    }

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)

# 결과 저장
print("\n" + "="*80)
print("모든 모델 학습 완료!")
print("="*80)

# 상세 결과를 텍스트 파일로 저장
with open('model_comparison_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("2022년 9월 신고량 예측 모델 비교 실험 결과\n")
    f.write("="*80 + "\n\n")

    f.write("[전체 모델 성능 비교]\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    # 각 역별 최고 모델
    f.write("[각 역별 최고 성능 모델]\n\n")
    for station in target_cols:
        station_data = results_df[results_df['Station'] == station]
        best = station_data.loc[station_data['MAE'].idxmin()]
        f.write(f"{station}:\n")
        f.write(f"  최고 모델: {best['Model']}\n")
        f.write(f"  MAE: {best['MAE']:.4f}\n")
        f.write(f"  RMSE: {best['RMSE']:.4f}\n")
        f.write(f"  R²: {best['R2']:.4f}\n")
        f.write(f"  예측 정확도: {best['Accuracy']:.2f}%\n\n")

    # 모델별 평균 성능
    f.write("[모델별 평균 성능]\n\n")
    model_avg = results_df.groupby('Model')[['MAE', 'RMSE', 'R2', 'Accuracy']].mean()
    f.write(model_avg.to_string())
    f.write("\n\n")

    # 2022년 9월 예측 결과
    f.write("[2022년 9월 신고량 예측 결과]\n\n")
    for station in target_cols:
        f.write(f"\n{station} 역:\n")
        f.write(f"  최고 모델: {all_predictions[station]['best_model']}\n")
        pred_dict = all_predictions[station]['predictions']
        for model_name, pred_values in pred_dict.items():
            avg_pred = pred_values.mean()
            f.write(f"  {model_name}: 평균 {avg_pred:.2f}건\n")

print("\n상세 결과가 'model_comparison_results.txt' 파일에 저장되었습니다.")

# 시각화
print("\n시각화 생성 중...")

# 1. 모델별 MAE 비교
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Comparison by Station', fontsize=16, fontweight='bold')

for idx, station in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    station_data = results_df[results_df['Station'] == station].sort_values('MAE')

    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(station_data))]
    bars = ax.barh(station_data['Model'], station_data['MAE'], color=colors)

    ax.set_xlabel('MAE', fontsize=10)
    ax.set_title(f'{station} Station', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # 값 표시
    for i, (bar, mae, acc) in enumerate(zip(bars, station_data['MAE'], station_data['Accuracy'])):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {mae:.2f} ({acc:.1f}%)',
                va='center', fontsize=8)

# 마지막 subplot 제거
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('model_comparison_mae.png', dpi=300, bbox_inches='tight')
print("그래프 저장: model_comparison_mae.png")

# 2. 모델별 평균 성능 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Average Model Performance Across All Stations', fontsize=16, fontweight='bold')

model_avg = results_df.groupby('Model')[['MAE', 'RMSE', 'R2', 'Accuracy']].mean().sort_values('MAE')

metrics = ['MAE', 'RMSE', 'R2', 'Accuracy']
titles = ['Mean Absolute Error (Lower is Better)', 'Root Mean Squared Error (Lower is Better)',
          'R² Score (Higher is Better)', 'Prediction Accuracy % (Higher is Better)']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    data = model_avg[metric].sort_values(ascending=(metric in ['MAE', 'RMSE']))

    if metric in ['R2', 'Accuracy']:
        colors = ['#2ecc71' if val == data.max() else '#3498db' for val in data]
    else:
        colors = ['#2ecc71' if val == data.min() else '#3498db' for val in data]

    bars = ax.barh(data.index, data.values, color=colors)
    ax.set_xlabel(metric, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()

    for bar, val in zip(bars, data.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {val:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_average_performance.png', dpi=300, bbox_inches='tight')
print("그래프 저장: model_average_performance.png")

# 3. 예측 정확도 히트맵
fig, ax = plt.subplots(figsize=(12, 6))
pivot_accuracy = results_df.pivot(index='Model', columns='Station', values='Accuracy')
sns.heatmap(pivot_accuracy, annot=True, fmt='.1f', cmap='RdYlGn', center=85,
            cbar_kws={'label': 'Accuracy %'}, ax=ax)
ax.set_title('Prediction Accuracy Heatmap (%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
print("그래프 저장: model_accuracy_heatmap.png")

# 4. 2022년 9월 예측값 비교
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('September 2022 Predictions by Model', fontsize=16, fontweight='bold')

for idx, station in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    pred_dict = all_predictions[station]['predictions']

    # 각 모델의 평균 예측값
    model_names = []
    avg_predictions = []
    for model_name, pred_values in pred_dict.items():
        model_names.append(model_name)
        avg_predictions.append(pred_values.mean())

    colors = ['#e74c3c' if name == all_predictions[station]['best_model'] else '#3498db'
              for name in model_names]
    bars = ax.barh(model_names, avg_predictions, color=colors)

    ax.set_xlabel('Average Predicted Reports', fontsize=10)
    ax.set_title(f'{station} Station', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    for bar, val in zip(bars, avg_predictions):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {val:.1f}', va='center', fontsize=9)

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('september_2022_predictions.png', dpi=300, bbox_inches='tight')
print("그래프 저장: september_2022_predictions.png")

print("\n" + "="*80)
print("모든 작업 완료!")
print("="*80)
print("\n생성된 파일:")
print("  - model_comparison_results.txt: 상세 결과 리포트")
print("  - model_comparison_mae.png: 역별 MAE 비교")
print("  - model_average_performance.png: 모델별 평균 성능")
print("  - model_accuracy_heatmap.png: 정확도 히트맵")
print("  - september_2022_predictions.png: 2022년 9월 예측값")
