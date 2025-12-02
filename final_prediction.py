"""
최종 최적화 모델로 2022년 9월 신고량 예측
더 빠른 실행을 위해 베스트 파라미터 직접 사용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("최종 최적화 모델로 2022년 9월 신고량 정확 예측")
print("="*80)

# 데이터 로드
df = pd.read_csv('reg_predict112.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# 2022년 9월 데이터 분리
predict_data = df[(df['year'] == 2022) & (df['month'] == 9)].copy()
train_data = df[~((df['year'] == 2022) & (df['month'] == 9))].copy()

print(f"\n학습 데이터: {train_data.shape}")
print(f"예측 대상 (2022년 9월): {predict_data.shape}")

target_cols = ['seogang', 'gongdeok', 'worldcup', 'hongik', 'yonggang']
feature_cols = [col for col in df.columns if col not in ['date', 'year', 'month'] + target_cols]

# 결과 저장
all_results = []
september_predictions = {}

for target in target_cols:
    print(f"\n{'='*80}")
    print(f"타겟 역: {target}")
    print(f"{'='*80}")

    X_train = train_data[feature_cols]
    y_train = train_data[target]
    X_pred = predict_data[feature_cols]

    # 학습/검증 분리 (성능 평가용)
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    X_val_split = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val_split = y_train[split_idx:]

    models_performance = {}
    models_predictions = {}

    # 1. CatBoost (최적화된 파라미터)
    print("\n[1] CatBoost")
    cat = cb.CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0
    )
    cat.fit(X_train_split, y_train_split)
    cat_pred_val = cat.predict(X_val_split)
    cat_mae = mean_absolute_error(y_val_split, cat_pred_val)
    cat_rmse = np.sqrt(mean_squared_error(y_val_split, cat_pred_val))
    cat_r2 = r2_score(y_val_split, cat_pred_val)
    cat_accuracy = 100 * (1 - cat_mae / y_val_split.mean())
    print(f"  검증 성능 - MAE: {cat_mae:.4f}, RMSE: {cat_rmse:.4f}, R²: {cat_r2:.4f}, 정확도: {cat_accuracy:.2f}%")

    # 전체 데이터로 재학습
    cat.fit(X_train, y_train)
    cat_pred = cat.predict(X_pred)
    models_performance['CatBoost'] = {'mae': cat_mae, 'rmse': cat_rmse, 'r2': cat_r2, 'accuracy': cat_accuracy}
    models_predictions['CatBoost'] = cat_pred

    # 2. LightGBM (최적화된 파라미터)
    print("\n[2] LightGBM")
    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=70,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train_split, y_train_split)
    lgbm_pred_val = lgbm.predict(X_val_split)
    lgbm_mae = mean_absolute_error(y_val_split, lgbm_pred_val)
    lgbm_rmse = np.sqrt(mean_squared_error(y_val_split, lgbm_pred_val))
    lgbm_r2 = r2_score(y_val_split, lgbm_pred_val)
    lgbm_accuracy = 100 * (1 - lgbm_mae / y_val_split.mean())
    print(f"  검증 성능 - MAE: {lgbm_mae:.4f}, RMSE: {lgbm_rmse:.4f}, R²: {lgbm_r2:.4f}, 정확도: {lgbm_accuracy:.2f}%")

    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_pred)
    models_performance['LightGBM'] = {'mae': lgbm_mae, 'rmse': lgbm_rmse, 'r2': lgbm_r2, 'accuracy': lgbm_accuracy}
    models_predictions['LightGBM'] = lgbm_pred

    # 3. XGBoost (최적화된 파라미터)
    print("\n[3] XGBoost")
    xgbm = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    xgbm.fit(X_train_split, y_train_split)
    xgbm_pred_val = xgbm.predict(X_val_split)
    xgbm_mae = mean_absolute_error(y_val_split, xgbm_pred_val)
    xgbm_rmse = np.sqrt(mean_squared_error(y_val_split, xgbm_pred_val))
    xgbm_r2 = r2_score(y_val_split, xgbm_pred_val)
    xgbm_accuracy = 100 * (1 - xgbm_mae / y_val_split.mean())
    print(f"  검증 성능 - MAE: {xgbm_mae:.4f}, RMSE: {xgbm_rmse:.4f}, R²: {xgbm_r2:.4f}, 정확도: {xgbm_accuracy:.2f}%")

    xgbm.fit(X_train, y_train)
    xgbm_pred = xgbm.predict(X_pred)
    models_performance['XGBoost'] = {'mae': xgbm_mae, 'rmse': xgbm_rmse, 'r2': xgbm_r2, 'accuracy': xgbm_accuracy}
    models_predictions['XGBoost'] = xgbm_pred

    # 4. Random Forest (최적화된 파라미터)
    print("\n[4] Random Forest")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_split, y_train_split)
    rf_pred_val = rf.predict(X_val_split)
    rf_mae = mean_absolute_error(y_val_split, rf_pred_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val_split, rf_pred_val))
    rf_r2 = r2_score(y_val_split, rf_pred_val)
    rf_accuracy = 100 * (1 - rf_mae / y_val_split.mean())
    print(f"  검증 성능 - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}, 정확도: {rf_accuracy:.2f}%")

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_pred)
    models_performance['Random Forest'] = {'mae': rf_mae, 'rmse': rf_rmse, 'r2': rf_r2, 'accuracy': rf_accuracy}
    models_predictions['Random Forest'] = rf_pred

    # 5. Gradient Boosting
    print("\n[5] Gradient Boosting")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb.fit(X_train_split, y_train_split)
    gb_pred_val = gb.predict(X_val_split)
    gb_mae = mean_absolute_error(y_val_split, gb_pred_val)
    gb_rmse = np.sqrt(mean_squared_error(y_val_split, gb_pred_val))
    gb_r2 = r2_score(y_val_split, gb_pred_val)
    gb_accuracy = 100 * (1 - gb_mae / y_val_split.mean())
    print(f"  검증 성능 - MAE: {gb_mae:.4f}, RMSE: {gb_rmse:.4f}, R²: {gb_r2:.4f}, 정확도: {gb_accuracy:.2f}%")

    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_pred)
    models_performance['Gradient Boosting'] = {'mae': gb_mae, 'rmse': gb_rmse, 'r2': gb_r2, 'accuracy': gb_accuracy}
    models_predictions['Gradient Boosting'] = gb_pred

    # 6. Stacking Ensemble (가중 평균 - 성능 기반)
    print("\n[6] Stacking Ensemble")
    val_preds = [cat_pred_val, lgbm_pred_val, xgbm_pred_val, rf_pred_val, gb_pred_val]
    final_preds = [cat_pred, lgbm_pred, xgbm_pred, rf_pred, gb_pred]

    # 각 모델의 MAE에 반비례하는 가중치
    maes = [cat_mae, lgbm_mae, xgbm_mae, rf_mae, gb_mae]
    weights = 1 / (np.array(maes) + 1e-10)
    weights = weights / weights.sum()

    ensemble_pred_val = np.average(val_preds, axis=0, weights=weights)
    ensemble_mae = mean_absolute_error(y_val_split, ensemble_pred_val)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_pred_val))
    ensemble_r2 = r2_score(y_val_split, ensemble_pred_val)
    ensemble_accuracy = 100 * (1 - ensemble_mae / y_val_split.mean())

    print(f"  가중치: CatBoost={weights[0]:.3f}, LightGBM={weights[1]:.3f}, XGBoost={weights[2]:.3f}, RF={weights[3]:.3f}, GB={weights[4]:.3f}")
    print(f"  검증 성능 - MAE: {ensemble_mae:.4f}, RMSE: {ensemble_rmse:.4f}, R²: {ensemble_r2:.4f}, 정확도: {ensemble_accuracy:.2f}%")

    ensemble_pred = np.average(final_preds, axis=0, weights=weights)
    models_performance['Stacking Ensemble'] = {'mae': ensemble_mae, 'rmse': ensemble_rmse, 'r2': ensemble_r2, 'accuracy': ensemble_accuracy}
    models_predictions['Stacking Ensemble'] = ensemble_pred

    # 최고 성능 모델 선택
    best_model = min(models_performance.items(), key=lambda x: x[1]['mae'])
    print(f"\n★★★ 최고 성능 모델: {best_model[0]}")
    print(f"    MAE: {best_model[1]['mae']:.4f}")
    print(f"    RMSE: {best_model[1]['rmse']:.4f}")
    print(f"    R²: {best_model[1]['r2']:.4f}")
    print(f"    예측 정확도: {best_model[1]['accuracy']:.2f}%")

    # 결과 저장
    for model_name, perf in models_performance.items():
        all_results.append({
            'Station': target,
            'Model': model_name,
            'MAE': perf['mae'],
            'RMSE': perf['rmse'],
            'R2': perf['r2'],
            'Accuracy': perf['accuracy']
        })

    september_predictions[target] = {
        'best_model': best_model[0],
        'predictions': models_predictions[best_model[0]],
        'all_predictions': models_predictions
    }

# 결과 DataFrame
results_df = pd.DataFrame(all_results)

# 상세 결과 저장
with open('final_prediction_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("2022년 9월 신고량 예측 - 최종 최적화 모델 결과\n")
    f.write("="*100 + "\n\n")

    f.write("="*100 + "\n")
    f.write("1. 전체 모델 성능 비교\n")
    f.write("="*100 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("="*100 + "\n")
    f.write("2. 각 역별 최고 성능 모델\n")
    f.write("="*100 + "\n\n")
    for station in target_cols:
        station_data = results_df[results_df['Station'] == station]
        best = station_data.loc[station_data['MAE'].idxmin()]
        f.write(f"[{station} 역]\n")
        f.write(f"  최고 모델: {best['Model']}\n")
        f.write(f"  MAE (평균 절대 오차): {best['MAE']:.4f}건\n")
        f.write(f"  RMSE (평균 제곱근 오차): {best['RMSE']:.4f}건\n")
        f.write(f"  R² (결정계수): {best['R2']:.4f}\n")
        f.write(f"  예측 정확도: {best['Accuracy']:.2f}%\n")
        f.write("\n")

    f.write("="*100 + "\n")
    f.write("3. 모델별 평균 성능\n")
    f.write("="*100 + "\n\n")
    model_avg = results_df.groupby('Model')[['MAE', 'RMSE', 'R2', 'Accuracy']].mean().sort_values('MAE')
    f.write(model_avg.to_string())
    f.write("\n\n")

    # 최고 평균 성능 모델
    best_avg_model = model_avg.iloc[0]
    f.write(f"전체 역 평균 최고 성능 모델: {model_avg.index[0]}\n")
    f.write(f"  평균 MAE: {best_avg_model['MAE']:.4f}건\n")
    f.write(f"  평균 예측 정확도: {best_avg_model['Accuracy']:.2f}%\n\n")

    f.write("="*100 + "\n")
    f.write("4. 2022년 9월 일별 예측 결과 (최고 성능 모델 사용)\n")
    f.write("="*100 + "\n\n")

    pred_df = predict_data[['date']].copy()
    for station in target_cols:
        pred_df[station] = september_predictions[station]['predictions']

    f.write(pred_df.to_string(index=False))
    f.write("\n\n")

    f.write("="*100 + "\n")
    f.write("5. 2022년 9월 역별 평균 예측 신고량\n")
    f.write("="*100 + "\n\n")

    total_pred = 0
    for station in target_cols:
        avg_pred = september_predictions[station]['predictions'].mean()
        total_pred += avg_pred
        f.write(f"{station:12s}: {avg_pred:6.2f}건/일  (모델: {september_predictions[station]['best_model']})\n")

    f.write(f"\n전체 5개 역 합계: {total_pred:.2f}건/일\n")
    f.write(f"전체 9월 예측:    {total_pred * 30:.2f}건/월\n")

    f.write("\n" + "="*100 + "\n")
    f.write("6. 성능 평가 지표 설명\n")
    f.write("="*100 + "\n\n")
    f.write("MAE (Mean Absolute Error, 평균 절대 오차):\n")
    f.write("  - 예측값과 실제값의 차이의 절대값 평균\n")
    f.write("  - 값이 작을수록 좋음\n")
    f.write("  - 단위: 건수\n\n")

    f.write("RMSE (Root Mean Squared Error, 평균 제곱근 오차):\n")
    f.write("  - 예측값과 실제값의 차이를 제곱한 값의 평균에 루트\n")
    f.write("  - 값이 작을수록 좋음\n")
    f.write("  - 큰 오차에 더 민감\n")
    f.write("  - 단위: 건수\n\n")

    f.write("R² (R-squared, 결정계수):\n")
    f.write("  - 모델이 데이터의 분산을 얼마나 설명하는지\n")
    f.write("  - 0~1 사이 값 (1에 가까울수록 좋음)\n")
    f.write("  - 0.8 이상: 매우 좋음, 0.6~0.8: 좋음, 0.4~0.6: 보통\n\n")

    f.write("예측 정확도 (%):\n")
    f.write("  - 100 × (1 - MAE / 평균값)\n")
    f.write("  - 백분율로 표현한 예측 정확도\n")
    f.write("  - 값이 클수록 좋음 (100%에 가까울수록 정확)\n\n")

print("\n" + "="*80)
print("모든 작업 완료!")
print("="*80)
print("\n상세 결과가 'final_prediction_results.txt' 파일에 저장되었습니다.")

# 시각화
print("\n시각화 생성 중...")

# 1. 역별 모델 성능 비교
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Final Optimized Model Performance by Station', fontsize=18, fontweight='bold', y=1.00)

for idx, station in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    station_data = results_df[results_df['Station'] == station].sort_values('MAE')

    colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(station_data))]
    bars = ax.barh(station_data['Model'], station_data['MAE'], color=colors)

    ax.set_xlabel('MAE (Mean Absolute Error)', fontsize=11, fontweight='bold')
    ax.set_title(f'{station.upper()} Station', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for bar, mae, acc in zip(bars, station_data['MAE'], station_data['Accuracy']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{mae:.2f} ({acc:.1f}%)', va='center', fontsize=9, fontweight='bold')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('final_model_performance.png', dpi=300, bbox_inches='tight')
print("✓ final_model_performance.png 저장 완료")

# 2. 모델별 평균 성능
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Average Model Performance Across All Stations', fontsize=18, fontweight='bold')

model_avg = results_df.groupby('Model')[['MAE', 'RMSE', 'R2', 'Accuracy']].mean()

metrics = [
    ('MAE', 'Mean Absolute Error (Lower is Better)', True, '#e74c3c'),
    ('RMSE', 'Root Mean Squared Error (Lower is Better)', True, '#e67e22'),
    ('R2', 'R² Score (Higher is Better)', False, '#27ae60'),
    ('Accuracy', 'Prediction Accuracy % (Higher is Better)', False, '#2ecc71')
]

for idx, (metric, title, lower_better, color) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    data = model_avg[metric].sort_values(ascending=lower_better)

    if lower_better:
        colors_list = [color if val == data.min() else '#3498db' for val in data]
    else:
        colors_list = [color if val == data.max() else '#3498db' for val in data]

    bars = ax.barh(data.index, data.values, color=colors_list)
    ax.set_xlabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, data.values):
        ax.text(bar.get_width() + (0.2 if metric != 'Accuracy' else 0.5),
                bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('final_average_performance.png', dpi=300, bbox_inches='tight')
print("✓ final_average_performance.png 저장 완료")

# 3. 정확도 히트맵
fig, ax = plt.subplots(figsize=(14, 7))
pivot_accuracy = results_df.pivot(index='Model', columns='Station', values='Accuracy')
sns.heatmap(pivot_accuracy, annot=True, fmt='.1f', cmap='RdYlGn', center=78,
            cbar_kws={'label': 'Accuracy (%)'}, ax=ax, linewidths=0.5)
ax.set_title('Prediction Accuracy Heatmap by Model and Station (%)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Station', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('final_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ final_accuracy_heatmap.png 저장 완료")

# 4. 2022년 9월 일별 예측값
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('September 2022 Daily Predictions (Best Model per Station)', fontsize=18, fontweight='bold', y=1.00)

for idx, station in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]
    dates = predict_data['date'].values
    predictions = september_predictions[station]['predictions']

    ax.plot(dates, predictions, marker='o', linewidth=2, markersize=6, color='#e74c3c', label='Predicted')
    ax.fill_between(dates, predictions, alpha=0.3, color='#e74c3c')

    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Reports', fontsize=11, fontweight='bold')
    ax.set_title(f'{station.upper()} - {september_predictions[station]["best_model"]}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # 평균값 표시
    avg_val = predictions.mean()
    ax.axhline(y=avg_val, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg: {avg_val:.1f}')
    ax.legend(loc='upper right')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('september_2022_daily_predictions.png', dpi=300, bbox_inches='tight')
print("✓ september_2022_daily_predictions.png 저장 완료")

# 5. 역별 9월 평균 예측량 비교
fig, ax = plt.subplots(figsize=(12, 7))
station_names = target_cols
avg_predictions = [september_predictions[s]['predictions'].mean() for s in target_cols]
colors_bar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

bars = ax.bar(station_names, avg_predictions, color=colors_bar, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Daily Reports in September 2022', fontsize=13, fontweight='bold')
ax.set_xlabel('Station', fontsize=13, fontweight='bold')
ax.set_title('September 2022 Average Predicted Reports by Station', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

for bar, val, station in zip(bars, avg_predictions, station_names):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}\n({september_predictions[station]["best_model"][:6]})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('september_average_by_station.png', dpi=300, bbox_inches='tight')
print("✓ september_average_by_station.png 저장 완료")

print("\n" + "="*80)
print("최종 결과 요약")
print("="*80)

# 요약 출력
for station in target_cols:
    best_model = september_predictions[station]['best_model']
    avg_pred = september_predictions[station]['predictions'].mean()
    station_perf = results_df[(results_df['Station'] == station) & (results_df['Model'] == best_model)].iloc[0]

    print(f"\n[{station.upper()}]")
    print(f"  최고 모델: {best_model}")
    print(f"  예측 정확도: {station_perf['Accuracy']:.2f}%")
    print(f"  9월 평균: {avg_pred:.2f}건/일")

total_avg = sum([september_predictions[s]['predictions'].mean() for s in target_cols])
print(f"\n전체 5개 역 9월 평균: {total_avg:.2f}건/일")
print(f"전체 9월 예측 총합: {total_avg * 30:.0f}건")

print("\n" + "="*80)
print("생성된 파일:")
print("="*80)
print("  [리포트]")
print("    - final_prediction_results.txt")
print("  [그래프]")
print("    - final_model_performance.png")
print("    - final_average_performance.png")
print("    - final_accuracy_heatmap.png")
print("    - september_2022_daily_predictions.png")
print("    - september_average_by_station.png")
print("="*80)
