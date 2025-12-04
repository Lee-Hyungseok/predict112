"""
NeuralProphet을 사용한 지구대별 신고건수 예측
- 타겟: 2022년 9월 신고건수 예측
- 각 지구대별로 모델 학습 및 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# MAPE 계산 함수
def calculate_mape(y_true, y_pred):
    """MAPE (Mean Absolute Percentage Error) 계산"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*100)
print("NeuralProphet을 사용한 지구대별 신고건수 예측")
print("="*100)

# 데이터 로드
df = pd.read_csv('reg_predict112.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n원본 데이터: {df.shape}")
print(f"데이터 기간: {df['date'].min()} ~ {df['date'].max()}")

# 2022년 9월 데이터 분리 (실제값)
target_date = '2022-09'
actual_data = df[df['date'].dt.to_period('M') == target_date].copy()
print(f"\n2022년 9월 실제 데이터: {len(actual_data)}일")

# 2022년 9월 이전 데이터로 학습
train_data = df[df['date'] < '2022-09-01'].copy()
print(f"학습 데이터: {train_data.shape[0]}개 샘플")
print(f"학습 기간: {train_data['date'].min()} ~ {train_data['date'].max()}")

# 지구대 목록
target_cols = ['seogang', 'gongdeok', 'worldcup', 'hongik', 'yonggang']

# 결과 저장용
results = {}
all_predictions = {}
evaluation_results = []

print("\n" + "="*100)
print("지구대별 NeuralProphet 모델 학습 및 예측")
print("="*100)

# 각 지구대별로 모델 학습 및 예측
for station in target_cols:
    print(f"\n{'='*100}")
    print(f"[{station.upper()} 지구대]")
    print(f"{'='*100}")

    # NeuralProphet용 데이터 준비 (ds, y 컬럼 필요)
    train_df = train_data[['date', station]].copy()
    train_df.columns = ['ds', 'y']

    # 실제값 준비
    actual_df = actual_data[['date', station]].copy()
    actual_values = actual_df[station].values

    print(f"학습 데이터 샘플 수: {len(train_df)}")
    print(f"학습 데이터 통계: 평균={train_df['y'].mean():.2f}, 최소={train_df['y'].min():.2f}, 최대={train_df['y'].max():.2f}")

    # NeuralProphet 모델 생성 및 학습
    model = NeuralProphet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=100,
        learning_rate=0.01,
        batch_size=32,
        loss_func='MSE',
        normalize='standardize'
    )

    print("\n모델 학습 중...")
    metrics = model.fit(train_df, freq='D')

    # 미래 30일 예측
    future = model.make_future_dataframe(train_df, periods=30, n_historic_predictions=False)
    forecast = model.predict(future)

    # 예측값 추출 (정확히 30일만)
    predictions = forecast['yhat1'].values[-30:]
    all_predictions[station] = predictions

    # 평가 지표 계산
    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    r2 = r2_score(actual_values, predictions)
    mape = calculate_mape(actual_values, predictions)
    accuracy = 100 * (1 - mae / actual_values.mean())

    # 예측 통계
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    pred_min = predictions.min()
    pred_max = predictions.max()
    pred_total = predictions.sum()

    actual_mean = actual_values.mean()
    actual_total = actual_values.sum()

    print(f"\n[평가 결과]")
    print(f"  MAE (평균 절대 오차): {mae:.4f}건")
    print(f"  RMSE (평균 제곱근 오차): {rmse:.4f}건")
    print(f"  R² (결정계수): {r2:.4f}")
    print(f"  MAPE (평균 절대 백분율 오차): {mape:.2f}%")
    print(f"  예측 정확도: {accuracy:.2f}%")

    print(f"\n[실제값 vs 예측값 비교]")
    print(f"  실제 일평균: {actual_mean:.2f}건/일")
    print(f"  예측 일평균: {pred_mean:.2f}건/일")
    print(f"  실제 9월 총합: {actual_total:.0f}건")
    print(f"  예측 9월 총합: {pred_total:.0f}건")
    print(f"  예측 오차: {pred_total - actual_total:+.0f}건 ({(pred_total - actual_total) / actual_total * 100:+.2f}%)")

    print(f"\n[예측값 통계]")
    print(f"  표준편차: {pred_std:.2f}건")
    print(f"  최소값: {pred_min:.2f}건")
    print(f"  최대값: {pred_max:.2f}건")

    # 결과 저장
    results[station] = {
        'actual': actual_values,
        'predicted': predictions,
        'dates': actual_df['date'].values
    }

    evaluation_results.append({
        'station': station,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'accuracy': accuracy,
        'actual_mean': actual_mean,
        'actual_total': actual_total,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pred_total': pred_total,
        'error': pred_total - actual_total,
        'error_pct': (pred_total - actual_total) / actual_total * 100
    })

# 전체 통계
total_actual = sum([r['actual_total'] for r in evaluation_results])
total_predicted = sum([r['pred_total'] for r in evaluation_results])
avg_mape = np.mean([r['mape'] for r in evaluation_results])
avg_r2 = np.mean([r['r2'] for r in evaluation_results])

print("\n" + "="*100)
print("전체 5개 지구대 통합 통계")
print("="*100)
print(f"실제 9월 총 신고량: {total_actual:.0f}건")
print(f"예측 9월 총 신고량: {total_predicted:.0f}건")
print(f"총 예측 오차: {total_predicted - total_actual:+.0f}건 ({(total_predicted - total_actual) / total_actual * 100:+.2f}%)")
print(f"평균 MAPE: {avg_mape:.2f}%")
print(f"평균 R²: {avg_r2:.4f}")

# 결과를 텍스트 파일로 저장
print("\n평가 결과를 NeuralProphet.txt 파일에 저장 중...")

with open('NeuralProphet.txt', 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("NeuralProphet을 사용한 지구대별 신고건수 예측 결과\n")
    f.write("="*100 + "\n\n")

    f.write("[데이터 정보]\n")
    f.write(f"학습 데이터: {len(train_data)}개 샘플\n")
    f.write(f"학습 기간: {train_data['date'].min().strftime('%Y-%m-%d')} ~ {train_data['date'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"예측 대상: 2022년 9월 (30일)\n")
    f.write(f"모델: NeuralProphet (n_lags=14, epochs=100, yearly/weekly seasonality)\n\n")

    f.write("="*100 + "\n")
    f.write("지구대별 상세 평가 결과\n")
    f.write("="*100 + "\n\n")

    for result in evaluation_results:
        f.write(f"[{result['station'].upper()} 지구대]\n")
        f.write(f"\n  <평가 지표>\n")
        f.write(f"    MAE (평균 절대 오차): {result['mae']:.4f}건\n")
        f.write(f"    RMSE (평균 제곱근 오차): {result['rmse']:.4f}건\n")
        f.write(f"    R² (결정계수): {result['r2']:.4f}\n")
        f.write(f"    MAPE (평균 절대 백분율 오차): {result['mape']:.2f}%\n")
        f.write(f"    예측 정확도: {result['accuracy']:.2f}%\n")

        f.write(f"\n  <실제값 vs 예측값>\n")
        f.write(f"    실제 일평균: {result['actual_mean']:.2f}건/일\n")
        f.write(f"    예측 일평균: {result['pred_mean']:.2f}건/일\n")
        f.write(f"    실제 9월 총합: {result['actual_total']:.0f}건\n")
        f.write(f"    예측 9월 총합: {result['pred_total']:.0f}건\n")
        f.write(f"    예측 오차: {result['error']:+.0f}건 ({result['error_pct']:+.2f}%)\n")

        f.write(f"\n  <예측값 통계>\n")
        f.write(f"    표준편차: {result['pred_std']:.2f}건\n")
        f.write(f"    최소값: {result['pred_min']:.2f}건\n")
        f.write(f"    최대값: {result['pred_max']:.2f}건\n")
        f.write("\n")

    f.write("="*100 + "\n")
    f.write("전체 통합 통계\n")
    f.write("="*100 + "\n\n")
    f.write(f"실제 9월 총 신고량: {total_actual:.0f}건\n")
    f.write(f"예측 9월 총 신고량: {total_predicted:.0f}건\n")
    f.write(f"총 예측 오차: {total_predicted - total_actual:+.0f}건 ({(total_predicted - total_actual) / total_actual * 100:+.2f}%)\n")
    f.write(f"평균 MAPE: {avg_mape:.2f}%\n")
    f.write(f"평균 R²: {avg_r2:.4f}\n\n")

    f.write("="*100 + "\n")
    f.write("2022년 9월 일별 실제값 vs 예측값\n")
    f.write("="*100 + "\n\n")

    # 일별 비교 테이블
    f.write(f"{'Date':<12}")
    for station in target_cols:
        f.write(f"{station.upper()}_actual  {station.upper()}_pred  ")
    f.write("\n")
    f.write("-"*150 + "\n")

    dates = actual_data['date'].values
    for i in range(len(dates)):
        date_str = pd.to_datetime(dates[i]).strftime('%Y-%m-%d')
        f.write(f"{date_str:<12}")
        for station in target_cols:
            actual_val = results[station]['actual'][i]
            pred_val = results[station]['predicted'][i]
            f.write(f"{actual_val:>6.0f}         {pred_val:>6.1f}      ")
        f.write("\n")

    f.write("\n" + "="*100 + "\n")
    f.write("성능 지표 설명\n")
    f.write("="*100 + "\n\n")
    f.write("MAE (Mean Absolute Error):\n")
    f.write("  예측값과 실제값의 차이의 절대값 평균. 값이 작을수록 좋음.\n\n")
    f.write("RMSE (Root Mean Squared Error):\n")
    f.write("  예측 오차의 제곱 평균에 루트. 큰 오차에 민감. 값이 작을수록 좋음.\n\n")
    f.write("R² (R-squared):\n")
    f.write("  모델이 데이터 분산을 설명하는 비율. -∞~1 사이 값. 1에 가까울수록 좋음.\n\n")
    f.write("MAPE (Mean Absolute Percentage Error):\n")
    f.write("  예측값과 실제값의 차이를 백분율로 나타낸 평균. 값이 작을수록 좋음.\n\n")
    f.write("예측 정확도 (%):\n")
    f.write("  100 × (1 - MAE / 평균값). 백분율 정확도. 100%에 가까울수록 좋음.\n\n")

    f.write("="*100 + "\n")
    f.write("결론\n")
    f.write("="*100 + "\n\n")
    f.write(f"NeuralProphet 모델을 사용하여 2022년 9월 신고건수를 예측한 결과,\n")
    f.write(f"평균 MAPE {avg_mape:.2f}%, 평균 R² {avg_r2:.4f}의 성능을 보였습니다.\n\n")
    f.write(f"전체 5개 지구대의 9월 총 신고량:\n")
    f.write(f"  - 실제: {total_actual:.0f}건\n")
    f.write(f"  - 예측: {total_predicted:.0f}건\n")
    f.write(f"  - 오차: {total_predicted - total_actual:+.0f}건 ({(total_predicted - total_actual) / total_actual * 100:+.2f}%)\n\n")

    f.write("지구대별 예측 성능 (MAPE 기준):\n")
    sorted_results = sorted(evaluation_results, key=lambda x: x['mape'])
    for rank, result in enumerate(sorted_results, 1):
        f.write(f"  {rank}. {result['station']:12s}: MAPE {result['mape']:6.2f}%, R² {result['r2']:.4f}\n")

print("✓ NeuralProphet.txt 파일 저장 완료")

# 시각화: 실제값 vs 예측값 비교
print("\n시각화 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Actual vs Predicted Reports by Station (September 2022) - NeuralProphet',
             fontsize=16, fontweight='bold', y=0.995)

for idx, station in enumerate(target_cols):
    ax = axes[idx // 3, idx % 3]

    dates = results[station]['dates']
    actual_vals = results[station]['actual']
    pred_vals = results[station]['predicted']

    # 실제값과 예측값 플롯 (실제값: 파란색, 예측값: 녹색)
    ax.plot(dates, actual_vals, marker='o', linewidth=2.5, markersize=8,
            color='blue', label='Actual', alpha=0.8)
    ax.plot(dates, pred_vals, marker='s', linewidth=2.5, markersize=7,
            color='green', label='Predicted', alpha=0.8, linestyle='--')

    # 통계 정보
    result = [r for r in evaluation_results if r['station'] == station][0]

    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Reports', fontsize=11, fontweight='bold')
    ax.set_title(f'{station.upper()} (MAPE: {result["mape"]:.2f}%, R²: {result["r2"]:.4f})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('NeuralProphet_predictions.png', dpi=300, bbox_inches='tight')
print("✓ NeuralProphet_predictions.png 저장 완료")

# 추가 시각화: MAPE 비교
fig, ax = plt.subplots(figsize=(12, 7))
stations_sorted = sorted(evaluation_results, key=lambda x: x['mape'])
station_names = [r['station'] for r in stations_sorted]
mapes = [r['mape'] for r in stations_sorted]

# 지구대별로 다른 색상 사용
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
bars = ax.bar(station_names, mapes, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Station', fontsize=13, fontweight='bold')
ax.set_title('NeuralProphet - MAPE by Station (Lower is Better)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

for bar, mape in zip(bars, mapes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{mape:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('NeuralProphet_mape_comparison.png', dpi=300, bbox_inches='tight')
print("✓ NeuralProphet_mape_comparison.png 저장 완료")

print("\n" + "="*100)
print("최종 요약")
print("="*100)
print(f"\n★ NeuralProphet 모델 예측 성능:")
print(f"   평균 MAPE: {avg_mape:.2f}%")
print(f"   평균 R²: {avg_r2:.4f}")
print(f"\n★ 2022년 9월 예측 결과:")
print(f"   실제 총합: {total_actual:.0f}건")
print(f"   예측 총합: {total_predicted:.0f}건")
print(f"   예측 오차: {total_predicted - total_actual:+.0f}건 ({(total_predicted - total_actual) / total_actual * 100:+.2f}%)")
print(f"\n생성된 파일:")
print(f"   - NeuralProphet.txt (상세 평가 결과)")
print(f"   - NeuralProphet_predictions.png (실제값 vs 예측값 그래프)")
print(f"   - NeuralProphet_mape_comparison.png (MAPE 비교 그래프)")
print("="*100)
