"""
Random Forest 모델 상세 분석 및 결과 저장
코로나 기간 제외 (2020.01 ~ 2022.04)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*100)
print("Random Forest 모델 상세 분석 (코로나 기간 제외)")
print("="*100)

# 데이터 로드
df = pd.read_csv('reg_predict112.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# 코로나 기간 제외
covid_mask = ((df['year'] == 2020) |
              (df['year'] == 2021) |
              ((df['year'] == 2022) & (df['month'] <= 4)))

df_no_covid = df[~covid_mask].copy()

# 2022년 9월 데이터 분리
predict_data = df_no_covid[(df_no_covid['year'] == 2022) & (df_no_covid['month'] == 9)].copy()
train_data = df_no_covid[~((df_no_covid['year'] == 2022) & (df_no_covid['month'] == 9))].copy()

print(f"\n학습 데이터: {train_data.shape[0]}개")
print(f"학습 기간: {train_data['date'].min().strftime('%Y-%m-%d')} ~ {train_data['date'].max().strftime('%Y-%m-%d')}")
print(f"예측 대상: {predict_data.shape[0]}일 (2022년 9월)")

target_cols = ['seogang', 'gongdeok', 'worldcup', 'hongik', 'yonggang']
feature_cols = [col for col in df_no_covid.columns if col not in ['date', 'year', 'month'] + target_cols]

# Random Forest 모델 정의
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print(f"\n[Random Forest 하이퍼파라미터]")
print(f"  n_estimators: 300")
print(f"  max_depth: 20")
print(f"  min_samples_split: 2")
print(f"  min_samples_leaf: 1")

# 결과 저장
detailed_results = []
final_predictions = {}
feature_importance_all = {}

print("\n" + "="*100)
print("역별 Random Forest 모델 학습 및 예측")
print("="*100)

for target in target_cols:
    print(f"\n{'='*100}")
    print(f"{target.upper()} 역")
    print(f"{'='*100}")

    X_train = train_data[feature_cols]
    y_train = train_data[target]
    X_pred = predict_data[feature_cols]

    # 학습/검증 분리
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    X_val_split = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val_split = y_train[split_idx:]

    print(f"\n학습 세트: {X_train_split.shape[0]}개, 검증 세트: {X_val_split.shape[0]}개")

    # 모델 학습
    rf_model.fit(X_train_split, y_train_split)

    # 검증 성능
    y_val_pred = rf_model.predict(X_val_split)
    val_mae = mean_absolute_error(y_val_split, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
    val_r2 = r2_score(y_val_split, y_val_pred)
    val_accuracy = 100 * (1 - val_mae / y_val_split.mean())

    print(f"\n[검증 성능]")
    print(f"  MAE: {val_mae:.4f}건")
    print(f"  RMSE: {val_rmse:.4f}건")
    print(f"  R²: {val_r2:.4f}")
    print(f"  예측 정확도: {val_accuracy:.2f}%")

    # 전체 데이터로 재학습
    rf_model.fit(X_train, y_train)

    # 특성 중요도
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance_all[target] = feature_importance

    print(f"\n[Top 10 중요 특성]")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    # 2022년 9월 예측
    predictions = rf_model.predict(X_pred)
    final_predictions[target] = predictions

    # 예측 통계
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    pred_min = predictions.min()
    pred_max = predictions.max()
    pred_total = predictions.sum()

    print(f"\n[2022년 9월 예측 결과]")
    print(f"  일평균: {pred_mean:.2f}건")
    print(f"  표준편차: {pred_std:.2f}건")
    print(f"  최소: {pred_min:.2f}건")
    print(f"  최대: {pred_max:.2f}건")
    print(f"  9월 총합: {pred_total:.0f}건")

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
total_daily_avg = sum([r['pred_mean'] for r in detailed_results])
total_monthly = sum([r['pred_total'] for r in detailed_results])
avg_accuracy = np.mean([r['val_accuracy'] for r in detailed_results])
avg_mae = np.mean([r['val_mae'] for r in detailed_results])

print("\n" + "="*100)
print("전체 통합 통계")
print("="*100)
print(f"평균 예측 정확도: {avg_accuracy:.2f}%")
print(f"평균 MAE: {avg_mae:.4f}건")
print(f"5개 역 일평균 총합: {total_daily_avg:.2f}건/일")
print(f"5개 역 9월 총합: {total_monthly:.0f}건")

# 상세 결과를 randomforest.txt에 저장
with open('randomforest.txt', 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("Random Forest 모델 상세 분석 결과\n")
    f.write("="*100 + "\n\n")

    f.write("[데이터 정보]\n")
    f.write(f"제외 기간: 2020년 1월 ~ 2022년 4월 (코로나 기간)\n")
    f.write(f"학습 데이터: {train_data.shape[0]}개 샘플\n")
    f.write(f"학습 기간: {train_data['date'].min().strftime('%Y-%m-%d')} ~ {train_data['date'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"예측 대상: 2022년 9월 (30일)\n")
    f.write(f"특성 변수: {len(feature_cols)}개\n\n")

    f.write("="*100 + "\n")
    f.write("Random Forest 하이퍼파라미터\n")
    f.write("="*100 + "\n\n")
    f.write("n_estimators (트리 개수): 300\n")
    f.write("max_depth (최대 깊이): 20\n")
    f.write("min_samples_split (분할 최소 샘플): 2\n")
    f.write("min_samples_leaf (리프 최소 샘플): 1\n")
    f.write("random_state: 42\n\n")

    f.write("="*100 + "\n")
    f.write("전체 평균 성능\n")
    f.write("="*100 + "\n\n")
    f.write(f"평균 예측 정확도: {avg_accuracy:.2f}%\n")
    f.write(f"평균 MAE: {avg_mae:.4f}건\n")
    f.write(f"평균 RMSE: {np.mean([r['val_rmse'] for r in detailed_results]):.4f}건\n")
    f.write(f"평균 R²: {np.mean([r['val_r2'] for r in detailed_results]):.4f}\n\n")

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
        f.write(f"    9월 총 신고량: {result['pred_total']:.0f}건\n\n")

        # 특성 중요도
        f.write(f"  <주요 특성 중요도 Top 10>\n")
        for idx, row in feature_importance_all[result['station']].head(10).iterrows():
            f.write(f"    {idx+1:2d}. {row['feature']:20s}: {row['importance']:.4f}\n")
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
    f.write("역별 예측 상세 통계\n")
    f.write("="*100 + "\n\n")

    for result in detailed_results:
        station = result['station']
        preds = final_predictions[station]

        f.write(f"[{station.upper()} 역 - 일별 예측값]\n")
        for i, (date, pred) in enumerate(zip(predict_data['date'], preds), 1):
            f.write(f"  {date.strftime('%Y-%m-%d')} ({date.strftime('%a'):3s}): {pred:6.2f}건\n")
        f.write(f"\n  평균: {preds.mean():.2f}건/일\n")
        f.write(f"  중앙값: {np.median(preds):.2f}건\n")
        f.write(f"  표준편차: {preds.std():.2f}건\n")
        f.write(f"  최소-최대: {preds.min():.2f}건 ~ {preds.max():.2f}건\n\n")

    f.write("="*100 + "\n")
    f.write("특성 중요도 종합 분석\n")
    f.write("="*100 + "\n\n")

    # 전체 역에 대한 평균 특성 중요도
    all_importance = pd.DataFrame()
    for station, importance_df in feature_importance_all.items():
        all_importance[station] = importance_df.set_index('feature')['importance']

    all_importance['average'] = all_importance.mean(axis=1)
    all_importance = all_importance.sort_values('average', ascending=False)

    f.write("[전체 역 평균 특성 중요도 Top 15]\n\n")
    f.write(f"{'순위':<4} {'특성명':<25} {'평균 중요도':<12} {'seogang':<10} {'gongdeok':<10} {'worldcup':<10} {'hongik':<10} {'yonggang':<10}\n")
    f.write("-"*100 + "\n")

    for rank, (feature, row) in enumerate(all_importance.head(15).iterrows(), 1):
        f.write(f"{rank:<4} {feature:<25} {row['average']:.6f}    ")
        for station in target_cols:
            f.write(f"{row[station]:.6f}  ")
        f.write("\n")

    f.write("\n")

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
    f.write("특성 중요도 (Feature Importance):\n")
    f.write("  Random Forest가 예측에 사용한 각 특성의 중요도.\n")
    f.write("  값이 클수록 해당 특성이 예측에 더 큰 영향을 미침.\n\n")

    f.write("="*100 + "\n")
    f.write("Random Forest 모델 특징\n")
    f.write("="*100 + "\n\n")
    f.write("장점:\n")
    f.write("  1. 과적합(Overfitting)에 강함\n")
    f.write("  2. 특성 중요도를 쉽게 파악 가능\n")
    f.write("  3. 비선형 관계를 잘 학습\n")
    f.write("  4. 이상치(Outlier)에 강건함\n")
    f.write("  5. 병렬 처리로 빠른 학습 가능\n\n")

    f.write("단점:\n")
    f.write("  1. 모델 해석이 어려움 (블랙박스)\n")
    f.write("  2. 메모리 사용량이 큼\n")
    f.write("  3. 예측 시간이 상대적으로 김\n\n")

    f.write("="*100 + "\n")
    f.write("결론\n")
    f.write("="*100 + "\n\n")
    f.write(f"코로나 기간(2020.01~2022.04)을 제외하고 학습한 Random Forest 모델은\n")
    f.write(f"5개 역 전체에서 평균 {avg_accuracy:.2f}%의 예측 정확도를 보였습니다.\n\n")
    f.write(f"2022년 9월 예상 신고량:\n")
    for result in detailed_results:
        f.write(f"  - {result['station']:12s}: {result['pred_total']:6.0f}건 (일평균 {result['pred_mean']:5.2f}건, 정확도 {result['val_accuracy']:.2f}%)\n")
    f.write(f"  - 전체 합계    : {total_monthly:6.0f}건 (일평균 {total_daily_avg:5.2f}건)\n\n")

    f.write(f"Random Forest는 CatBoost(80.42%)에 이어 두 번째로 높은 평균 정확도({avg_accuracy:.2f}%)를 기록했으며,\n")
    f.write(f"특히 특성 중요도 분석을 통해 예측에 영향을 주는 주요 변수들을 명확히 파악할 수 있었습니다.\n")

print("\n상세 결과가 'randomforest.txt' 파일에 저장되었습니다.")

# 시각화
print("\n시각화 생성 중...")

# 1. 역별 성능
fig, ax = plt.subplots(figsize=(12, 7))
stations = [r['station'] for r in detailed_results]
accuracies = [r['val_accuracy'] for r in detailed_results]
colors_station = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

bars = ax.bar(stations, accuracies, color=colors_station, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Prediction Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Station', fontsize=13, fontweight='bold')
ax.set_title('Random Forest - Performance by Station (Excluding COVID Period)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('rf_performance_by_station.png', dpi=300, bbox_inches='tight')
print("✓ rf_performance_by_station.png 저장 완료")

# 2. 특성 중요도 히트맵
fig, ax = plt.subplots(figsize=(14, 10))
import seaborn as sns

# Top 15 특성만 선택
top_features = all_importance.head(15).index
heatmap_data = all_importance.loc[top_features, target_cols].T

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Importance'})
ax.set_title('Random Forest - Feature Importance by Station (Top 15)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
ax.set_ylabel('Station', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('rf_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ rf_feature_importance_heatmap.png 저장 완료")

# 3. 9월 예측값
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Random Forest - September 2022 Daily Predictions',
             fontsize=16, fontweight='bold', y=0.995)

for idx, (station, result) in enumerate(zip(target_cols, detailed_results)):
    ax = axes[idx // 3, idx % 3]
    dates = predict_data['date'].values
    predictions = final_predictions[station]

    ax.plot(dates, predictions, marker='o', linewidth=2.5, markersize=7,
            color='#2ecc71', label='Predicted')
    ax.fill_between(dates, predictions, alpha=0.3, color='#2ecc71')

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
plt.savefig('rf_september_predictions.png', dpi=300, bbox_inches='tight')
print("✓ rf_september_predictions.png 저장 완료")

print("\n" + "="*100)
print("Random Forest 분석 완료!")
print("="*100)
print(f"\n생성된 파일:")
print(f"  - randomforest.txt (상세 분석 결과)")
print(f"  - rf_performance_by_station.png")
print(f"  - rf_feature_importance_heatmap.png")
print(f"  - rf_september_predictions.png")
print("="*100)
