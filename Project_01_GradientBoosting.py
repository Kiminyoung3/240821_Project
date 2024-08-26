import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
sns.set_theme(style='darkgrid')

# 1. 훈련 데이터 로드
train_df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 테스트 데이터 로드
test_df = pd.read_csv('./data/6.Heatstroke_patient_prediction_test_data.csv')

# 3. 새로운 헤더 설정_1) train
train_header = ['1날짜 및 시간', '2이송 인원', '3최고기온', '4평균기온', '5최저기온', '6일조시간', '7평균풍속(m/s)', '8평균운량', '9평균습도(%)',
                '10강수량합계(mm)', '11최소상대습도(%)', '12합계 전천 일사량(MJ/m^2)', '13평균 증기압(hPa)', '14평균 현지 기압(hPa)',
                '15평균 해면 기압(hPa)', '16최대풍속(m/s)', '17최대 순간 풍속(m/s)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '21월', '22요일',
                '23주말 및 공휴일',
                '24낮_맑음 비율', '25낮_흐림 비율', '26낮_비 비율', '27낮_번개 비율', '28밤_맑음 비율', '29밤_흐림 비율', '30밤_비 비율', '31밤_번개 있음',
                '32전일 최고 기온 차이', '33전일 평균 기온 차이', '34전일 최저 기온 차이', '35최고 기온 이동 평균(5일간)', '36평균 기온 이동 평균(5일간)',
                '37체감 온도 이동 평균(5일간)', '38불쾌지수 이동 평균(5일간)', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)', '41Year']

# 3. 새로운 헤더 설정_1) test(train 데이터와 비교했을때 21월, 41Year 컬럼이 없음)
test_header = ['1날짜 및 시간', '2이송 인원', '3최고기온', '4평균기온', '5최저기온', '6일조시간', '7평균풍속(m/s)', '8평균운량', '9평균습도(%)',
               '10강수량합계(mm)', '11최소상대습도(%)', '12합계 전천 일사량(MJ/m^2)', '13평균 증기압(hPa)', '14평균 현지 기압(hPa)',
               '15평균 해면 기압(hPa)', '16최대풍속(m/s)', '17최대 순간 풍속(m/s)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '22요일',
               '23주말 및 공휴일',
               '24낮_맑음 비율', '25낮_흐림 비율', '26낮_비 비율', '27낮_번개 비율', '28밤_맑음 비율', '29밤_흐림 비율', '30밤_비 비율', '31밤_번개 있음',
               '32전일 최고 기온 차이', '33전일 평균 기온 차이', '34전일 최저 기온 차이', '35최고 기온 이동 평균(5일간)', '36평균 기온 이동 평균(5일간)',
               '37체감 온도 이동 평균(5일간)', '38불쾌지수 이동 평균(5일간)', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)']

train_df.columns = train_header
test_df.columns = test_header


# 4. '1날짜 및 시간' 열 변환(연, 월, 일, 시간으로 분할)
def process_datetime(df):
    df['1날짜 및 시간'] = pd.to_datetime(df['1날짜 및 시간'], format='%m/%d/%Y %I:%M:%S %p')
    df['Year'] = df['1날짜 및 시간'].dt.year
    df['Month'] = df['1날짜 및 시간'].dt.month
    df['Day'] = df['1날짜 및 시간'].dt.day
    df['Hour'] = df['1날짜 및 시간'].dt.hour
    df.drop(columns=['1날짜 및 시간'], inplace=True)  # 원본 열 삭제


process_datetime(train_df)
process_datetime(test_df)

# 5. 데이터 전처리
# 훈련 데이터에서 필요한 열 선택
X_train = train_df[['Year', 'Month', 'Day', '3최고기온', '4평균기온', '9평균습도(%)',
                    '10강수량합계(mm)', '18최고-최저 기온차', '19체감온도', '20불쾌지수',
                    '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)']]
y_train = train_df['2이송 인원']

# 테스트 데이터에서 필요한 열 선택
X_test = test_df[['Year', 'Month', 'Day', '3최고기온', '4평균기온', '9평균습도(%)',
                  '10강수량합계(mm)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)']]
y_test = test_df['2이송 인원']

# 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. KFold 교차검증 설정 및 모델 훈련
# 데이터를 5개의 폴드로 나누고 각 폴드를 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_list = []
rmse_list = []
mae_list = []
r2_list = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_fold, y_train_fold)

    y_val_pred = model.predict(X_val_fold)

    mse_list.append(mean_squared_error(y_val_fold, y_val_pred))
    rmse_list.append(np.sqrt(mse_list[-1]))
    mae_list.append(mean_absolute_error(y_val_fold, y_val_pred))
    r2_list.append(r2_score(y_val_fold, y_val_pred))

# 최종 평균 출력
print("-----Gradient Boosting KFold 교차검증 결과-----")
print(f'KFold Cross-Validation MSE: {np.mean(mse_list):.2f}')
print(f'KFold Cross-Validation RMSE: {np.mean(rmse_list):.2f}')
print(f'KFold Cross-Validation MAE: {np.mean(mae_list):.2f}')
print(f'KFold Cross-Validation R^2: {np.mean(r2_list):.2f}')
print('\n')

# 7. 전체 훈련 데이터로 모델 훈련
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. 예측
y_test_pred = model.predict(X_test)

# 9. 성능 평가
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("-----Gradient Boosting 성능평가 결과-----")
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R^2: {r2:.2f}')

# 10. 결과 저장
test_df['patient prediction'] = y_test_pred
test_df.to_csv('./result/6.Heatstroke_test_predictions_GradientBoosting.csv', index=False)

# 11. 예측 결과 시각화
plt.figure(figsize=(18, 12))

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False    # 음수 기호 깨짐 방지

# 필터링된 데이터 생성 (값이 20 이상 160 이하인 것만 포함)
filtered_indices = [i for i in range(len(y_test)) if 20 <= y_test[i] <= 160 and 20 <= y_test_pred[i] <= 160]
filtered_actual = [y_test[i] for i in filtered_indices]
filtered_predicted = [y_test_pred[i] for i in filtered_indices]
filtered_errors = [y_test[i] - y_test_pred[i] for i in filtered_indices]

# 색상 설정
color_actual = 'orange'
color_predicted = '#2ca02c'  # Green
color_error = '#9467bd'  # Purple
color_diagonal = 'orange'  # (Same as actual values for consistency)

# 11-1. 산점도 그래프
plt.subplot(2, 2, 1)
plt.scatter(filtered_actual, filtered_predicted, alpha=0.5, color=color_predicted)
# 대각선 그리기
plt.plot([min(filtered_actual), max(filtered_actual)], [min(filtered_actual), max(filtered_actual)],
         linestyle='--', color=color_diagonal, linewidth=2)
plt.xlabel('실제 열사병 이송 환자 수')
plt.ylabel('예측 열사병 이송 환자 수')
plt.title('Gradient Boosting 회귀 모델에 따른 실제와 예측 환자 수 비교1')

# 11-2. 실제 값과 예측 값의 분포를 비교하는 히스토그램
plt.subplot(2, 2, 2)
sns.histplot(filtered_actual, color=color_actual, label='실제 값', kde=True, stat='density', linewidth=0, bins=30)
sns.histplot(filtered_predicted, color=color_predicted, label='예측 값', kde=True, stat='density', linewidth=0, bins=30)
plt.xlabel('열사병 이송 환자 수')
plt.ylabel('밀도')
plt.title('Gradient Boosting 회귀 모델에 따른 실제와 예측 환자 수 비교2')
plt.legend()

# 11-3. 실제 값과 예측 값의 시간에 따른 변화 (선 그래프)
plt.subplot(2, 2, 3)
plt.plot(range(len(filtered_actual)), filtered_actual, label='실제 값', linestyle='-', marker='o', color=color_actual)
plt.plot(range(len(filtered_predicted)), filtered_predicted, label='예측 값', linestyle='--', marker='x', color=color_predicted)
plt.xlabel('인덱스')
plt.ylabel('열사병 이송 환자 수')
plt.title('시간에 따른 실제 값과 예측 값')
plt.legend()

# 11-4. 오차 분포의 히스토그램
plt.subplot(2, 2, 4)
sns.histplot(filtered_errors, bins=30, kde=True, color=color_error)
plt.xlabel('오차')
plt.ylabel('빈도')
plt.title('오차 분포 (히스토그램)')


plt.tight_layout()
plt.savefig('./result/GradientBoosting_result.png')
plt.show()