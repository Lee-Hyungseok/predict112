"""
코로나 기간 제외 + 단일 최고 성능 모델로 통합 예측
- 제외 기간: 2020년 1월 ~ 2022년 4월 (코로나 기간)
- 하나의 모델로 5개 역 모두 평가
- 평균 성능이 가장 높은 모델만 선택하여 상세 분석
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

print("="*100)
print("코로나 기간 제외 + 최고 성능 단일 모델 선택")
print("="*100)

# 데이터 로드
df = pd.read_csv('reg_predict112.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

print(f"\n원본 데이터: {df.shape}")

# 코로나 기간 제외 (2020-01 ~ 2022-04)
covid_mask = ((df['year'] == 2020) |
              (df['year'] == 2021) |
              ((df['year'] == 2022) & (df['month'] <= 4)))

df_no_covid = df[~covid_mask].copy()
print(f"코로나 기간 제외 후: {df_no_covid.shape}")
print(f"제외된 데이터: {covid_mask.sum()}개")

# 2022년 9월 데이터 분리
predict_data = df_no_covid[(df_no_covid['year'] == 2022) & (df_no_covid['month'] == 9)].copy()
train_data = df_no_covid[~((df_no_covid['year'] == 2022) & (df_no_covid['month'] == 9))].copy()

print(f"\n학습 데이터: {train_data.shape}")
print(f"예측 대상 (2022년 9월): {predict_data.shape}")
print(f"학습 데이터 기간: {train_data['date'].min()} ~ {train_data['date'].max()}")

target_cols = ['seogang', 'gongdeok', 'worldcup', 'hongik', 'yonggang']
feature_cols = [col for col in df_no_covid.columns if col not in ['date', 'year', 'month'] + target_cols]

print(f"\n특성 변수: {len(feature_cols)}개")

# 모델 정의
models = {
    'CatBoost': cb.CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=70,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}

# 각 모델의 전체 평균 성능 평가
model_avg_performance = {}

print("\n" + "="*100)
print("모델별 5개 역 통합 성능 평가")
print("="*100)

for model_name, model in models.items():
    print(f"\n{'='*100}")
    print(f"모델: {model_name}")
    print(f"{'='*100}")

    station_performances = []

    for target in target_cols:
        X_train = train_data[feature_cols]
        y_train = train_data[target]
        X_pred = predict_data[feature_cols]

        # 학습/검증 분리
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train[:split_idx]
        X_val_split = X_train[split_idx:]
        y_train_split = y_train[:split_idx]
        y_val_split = y_train[split_idx:]

        # 모델 학습 및 평가
        if 'Neural' in model_name:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_val_scaled = scaler.transform(X_val_split)
            model.fit(X_train_scaled, y_train_split)
            y_pred = model.predict(X_val_scaled)
        else:
            model.fit(X_train_split, y_train_split)
            y_pred = model.predict(X_val_split)

        mae = mean_absolute_error(y_val_split, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
        r2 = r2_score(y_val_split, y_pred)
        accuracy = 100 * (1 - mae / y_val_split.mean())

        station_performances.append({
            'station': target,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy
        })

        print(f"  {target:12s}: MAE={mae:6.2f}, RMSE={rmse:6.2f}, R²={r2:6.4f}, 정확도={accuracy:6.2f}%")

    # 평균 성능 계산
    avg_mae = np.mean([p['mae'] for p in station_performances])
    avg_rmse = np.mean([p['rmse'] for p in station_performances])
    avg_r2 = np.mean([p['r2'] for p in station_performances])
    avg_accuracy = np.mean([p['accuracy'] for p in station_performances])

    model_avg_performance[model_name] = {
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2,
        'avg_accuracy': avg_accuracy,
        'station_details': station_performances
    }

    print(f"\n  [평균 성능]")
    print(f"  평균 MAE: {avg_mae:.4f}")
    print(f"  평균 RMSE: {avg_rmse:.4f}")
    print(f"  평균 R²: {avg_r2:.4f}")
    print(f"  평균 정확도: {avg_accuracy:.2f}%")

# 최고 성능 모델 선택
best_model_name = max(model_avg_performance.items(), key=lambda x: x[1]['avg_accuracy'])[0]
best_model_info = model_avg_performance[best_model_name]

print("\n" + "="*100)
print(f"★★★ 최고 성능 모델: {best_model_name}")
print("="*100)
print(f"평균 MAE: {best_model_info['avg_mae']:.4f}")
print(f"평균 RMSE: {best_model_info['avg_rmse']:.4f}")
print(f"평균 R²: {best_model_info['avg_r2']:.4f}")
print(f"평균 정확도: {best_model_info['avg_accuracy']:.2f}%")

# 최고 모델로 최종 예측 수행
print("\n" + "="*100)
print(f"{best_model_name} 모델로 2022년 9월 최종 예측")
print("="*100)

best_model = models[best_model_name]
final_predictions = {}
detailed_results = []

for target in target_cols:
    print(f"\n[{target} 역]")

    X_train = train_data[feature_cols]
    y_train = train_data[target]
    X_pred = predict_data[feature_cols]

    # 학습/검증 분리 (성능 재확인)
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    X_val_split = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val_split = y_train[split_idx:]

    # 검증 성능
    best_model.fit(X_train_split, y_train_split)
    y_val_pred = best_model.predict(X_val_split)

    val_mae = mean_absolute_error(y_val_split, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
    val_r2 = r2_score(y_val_split, y_val_pred)
    val_accuracy = 100 * (1 - val_mae / y_val_split.mean())

    print(f"  검증 성능: MAE={val_mae:.2f}, RMSE={val_rmse:.2f}, R²={val_r2:.4f}, 정확도={val_accuracy:.2f}%")

    # 전체 데이터로 재학습 후 예측
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_pred)

    final_predictions[target] = predictions

    # 예측 통계
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    pred_min = predictions.min()
    pred_max = predictions.max()
    pred_total = predictions.sum()

    print(f"  9월 예측 통계:")
    print(f"    평균: {pred_mean:.2f}건/일")
    print(f"    표준편차: {pred_std:.2f}")
    print(f"    최소: {pred_min:.2f}건")
    print(f"    최대: {pred_max:.2f}건")
    print(f"    9월 총합: {pred_total:.0f}건")

    detailed_results.append({
        'station': target,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_accuracy': val_accuracy,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pred_total': pred_total
    })

# 전체 통계
total_daily_avg = sum([p['pred_mean'] for p in detailed_results])
total_monthly = sum([p['pred_total'] for p in detailed_results])

print("\n" + "="*100)
print("전체 5개 역 통합 통계")
print("="*100)
print(f"일평균 총 신고량: {total_daily_avg:.2f}건/일")
print(f"9월 총 신고량: {total_monthly:.0f}건")

# 상세 결과 저장
with open('finalmodel.txt', 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("코로나 기간 제외 후 최고 성능 단일 모델 분석 결과\n")
    f.write("="*100 + "\n\n")

    f.write("[데이터 정보]\n")
    f.write(f"제외 기간: 2020년 1월 ~ 2022년 4월 (코로나 기간)\n")
    f.write(f"학습 데이터: {train_data.shape[0]}개 샘플\n")
    f.write(f"학습 기간: {train_data['date'].min().strftime('%Y-%m-%d')} ~ {train_data['date'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"예측 대상: 2022년 9월 (30일)\n")
    f.write(f"특성 변수: {len(feature_cols)}개\n\n")

    f.write("="*100 + "\n")
    f.write("모델별 평균 성능 비교\n")
    f.write("="*100 + "\n\n")

    # 성능순 정렬
    sorted_models = sorted(model_avg_performance.items(),
                          key=lambda x: x[1]['avg_accuracy'],
                          reverse=True)

    f.write(f"{'순위':<4} {'모델명':<20} {'평균 MAE':<12} {'평균 RMSE':<12} {'평균 R²':<12} {'평균 정확도':<12}\n")
    f.write("-"*100 + "\n")

    for rank, (model_name, perf) in enumerate(sorted_models, 1):
        marker = " ★★★" if model_name == best_model_name else ""
        f.write(f"{rank:<4} {model_name:<20} {perf['avg_mae']:<12.4f} {perf['avg_rmse']:<12.4f} "
                f"{perf['avg_r2']:<12.4f} {perf['avg_accuracy']:<12.2f}%{marker}\n")

    f.write("\n" + "="*100 + "\n")
    f.write(f"최고 성능 모델: {best_model_name}\n")
    f.write("="*100 + "\n\n")

    f.write(f"[전체 평균 성능]\n")
    f.write(f"평균 MAE: {best_model_info['avg_mae']:.4f}건\n")
    f.write(f"평균 RMSE: {best_model_info['avg_rmse']:.4f}건\n")
    f.write(f"평균 R²: {best_model_info['avg_r2']:.4f}\n")
    f.write(f"평균 정확도: {best_model_info['avg_accuracy']:.2f}%\n\n")

    f.write("="*100 + "\n")
    f.write("역별 상세 성능 및 예측 결과\n")
    f.write("="*100 + "\n\n")

    for result in detailed_results:
        f.write(f"[{result['station'].upper()} 역]\n")
        f.write(f"\n  <검증 성능>\n")
        f.write(f"    MAE (평균 절대 오차): {result['val_mae']:.4f}건\n")
        f.write(f"    RMSE (평균 제곱근 오차): {result['val_rmse']:.4f}건\n")
        f.write(f"    R² (결정계수): {result['val_r2']:.4f}\n")
        f.write(f"    예측 정확도: {result['val_accuracy']:.2f}%\n")
        f.write(f"\n  <2022년 9월 예측 결과>\n")
        f.write(f"    일평균 신고량: {result['pred_mean']:.2f}건/일\n")
        f.write(f"    표준편차: {result['pred_std']:.2f}건\n")
        f.write(f"    최소값: {result['pred_min']:.2f}건\n")
        f.write(f"    최대값: {result['pred_max']:.2f}건\n")
        f.write(f"    9월 총 신고량: {result['pred_total']:.0f}건\n")
        f.write("\n")

    f.write("="*100 + "\n")
    f.write("전체 통합 통계\n")
    f.write("="*100 + "\n\n")
    f.write(f"5개 역 합계 일평균: {total_daily_avg:.2f}건/일\n")
    f.write(f"5개 역 합계 9월 총합: {total_monthly:.0f}건\n\n")

    f.write("="*100 + "\n")
    f.write("2022년 9월 일별 예측값 (30일)\n")
    f.write("="*100 + "\n\n")

    pred_df = predict_data[['date']].copy()
    for station in target_cols:
        pred_df[station] = final_predictions[station]
    pred_df['total'] = pred_df[target_cols].sum(axis=1)

    f.write(pred_df.to_string(index=False))
    f.write("\n\n")

    f.write("="*100 + "\n")
    f.write("성능 지표 설명\n")
    f.write("="*100 + "\n\n")
    f.write("MAE (Mean Absolute Error):\n")
    f.write("  예측값과 실제값의 차이의 절대값 평균. 값이 작을수록 좋음.\n\n")
    f.write("RMSE (Root Mean Squared Error):\n")
    f.write("  예측 오차의 제곱 평균에 루트. 큰 오차에 민감. 값이 작을수록 좋음.\n\n")
    f.write("R² (R-squared):\n")
    f.write("  모델이 데이터 분산을 설명하는 비율. 0~1 사이 값. 1에 가까울수록 좋음.\n\n")
    f.write("예측 정확도 (%):\n")
    f.write("  100 × (1 - MAE / 평균값). 백분율 정확도. 100%에 가까울수록 좋음.\n\n")

    f.write("="*100 + "\n")
    f.write("결론\n")
    f.write("="*100 + "\n\n")
    f.write(f"코로나 기간(2020.01~2022.04)을 제외하고 학습한 결과,\n")
    f.write(f"{best_model_name} 모델이 5개 역 전체에서 평균 {best_model_info['avg_accuracy']:.2f}%의\n")
    f.write(f"예측 정확도로 최고 성능을 보였습니다.\n\n")
    f.write(f"2022년 9월 예상 신고량:\n")
    for result in detailed_results:
        f.write(f"  - {result['station']:12s}: {result['pred_total']:6.0f}건 (일평균 {result['pred_mean']:5.2f}건)\n")
    f.write(f"  - 전체 합계    : {total_monthly:6.0f}건 (일평균 {total_daily_avg:5.2f}건)\n")

print("\n상세 결과가 'finalmodel.txt' 파일에 저장되었습니다.")

# 시각화
print("\n시각화 생성 중...")

# 1. 모델 성능 비교
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance Comparison (Excluding COVID Period)', fontsize=16, fontweight='bold')

# 평균 정확도 비교
ax1 = axes[0]
sorted_models_data = sorted(model_avg_performance.items(),
                            key=lambda x: x[1]['avg_accuracy'],
                            reverse=True)
model_names = [m[0] for m in sorted_models_data]
accuracies = [m[1]['avg_accuracy'] for m in sorted_models_data]
colors = ['#e74c3c' if name == best_model_name else '#3498db' for name in model_names]

bars = ax1.barh(model_names, accuracies, color=colors)
ax1.set_xlabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Average Prediction Accuracy', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', va='center', fontsize=11, fontweight='bold')

# 평균 MAE 비교
ax2 = axes[1]
maes = [m[1]['avg_mae'] for m in sorted_models_data]
colors2 = ['#e74c3c' if name == best_model_name else '#3498db' for name in model_names]

bars2 = ax2.barh(model_names, maes, color=colors2)
ax2.set_xlabel('Average MAE', fontsize=12, fontweight='bold')
ax2.set_title('Average Mean Absolute Error (Lower is Better)', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

for bar, mae in zip(bars2, maes):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{mae:.2f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('final_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ final_model_comparison.png 저장 완료")

# 2. 최고 모델의 역별 성능
fig, ax = plt.subplots(figsize=(12, 7))
stations = [r['station'] for r in detailed_results]
accuracies_by_station = [r['val_accuracy'] for r in detailed_results]
colors_station = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

bars = ax.bar(stations, accuracies_by_station, color=colors_station, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Prediction Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Station', fontsize=13, fontweight='bold')
ax.set_title(f'{best_model_name} - Performance by Station (Excluding COVID Period)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

for bar, acc in zip(bars, accuracies_by_station):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('final_best_model_by_station.png', dpi=300, bbox_inches='tight')
print("✓ final_best_model_by_station.png 저장 완료")

# 3. 9월 예측값
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'September 2022 Predictions - {best_model_name} (Excluding COVID Period)',
             fontsize=16, fontweight='bold', y=0.995)

for idx, (station, result) in enumerate(zip(target_cols, detailed_results)):
    ax = axes[idx // 3, idx % 3]
    dates = predict_data['date'].values
    predictions = final_predictions[station]

    ax.plot(dates, predictions, marker='o', linewidth=2.5, markersize=7,
            color='#e74c3c', label='Predicted')
    ax.fill_between(dates, predictions, alpha=0.3, color='#e74c3c')

    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Reports', fontsize=11, fontweight='bold')
    ax.set_title(f'{station.upper()} (Avg: {result["pred_mean"]:.1f}, Accuracy: {result["val_accuracy"]:.1f}%)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    avg_line = ax.axhline(y=result['pred_mean'], color='blue', linestyle='--',
                          linewidth=2, alpha=0.7)
    ax.legend([avg_line], [f'Average: {result["pred_mean"]:.1f}'], loc='upper right')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('final_september_predictions.png', dpi=300, bbox_inches='tight')
print("✓ final_september_predictions.png 저장 완료")

# 4. 역별 9월 총 예측량
fig, ax = plt.subplots(figsize=(12, 7))
totals = [r['pred_total'] for r in detailed_results]
colors_total = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

bars = ax.bar(stations, totals, color=colors_total, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Total Reports in September 2022', fontsize=13, fontweight='bold')
ax.set_xlabel('Station', fontsize=13, fontweight='bold')
ax.set_title(f'September 2022 Total Predictions by Station - {best_model_name}',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

for bar, total, result in zip(bars, totals, detailed_results):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{int(total)}\n({result["pred_mean"]:.1f}/day)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('final_total_by_station.png', dpi=300, bbox_inches='tight')
print("✓ final_total_by_station.png 저장 완료")

print("\n" + "="*100)
print("최종 요약")
print("="*100)
print(f"\n★ 최고 성능 모델: {best_model_name}")
print(f"   평균 예측 정확도: {best_model_info['avg_accuracy']:.2f}%")
print(f"   평균 MAE: {best_model_info['avg_mae']:.4f}건")
print(f"\n★ 2022년 9월 전체 예측: {total_monthly:.0f}건 (일평균 {total_daily_avg:.2f}건)")
print(f"\n생성된 파일:")
print(f"   - finalmodel.txt (상세 분석 결과)")
print(f"   - final_model_comparison.png")
print(f"   - final_best_model_by_station.png")
print(f"   - final_september_predictions.png")
print(f"   - final_total_by_station.png")
print("="*100)
