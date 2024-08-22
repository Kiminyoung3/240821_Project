import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
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


# 4. 날짜 및 시간 열 변환
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
X_train = train_df[['Year', 'Month', 'Day', 'Hour', '3최고기온', '4평균기온', '9평균습도(%)',
                    '10강수량합계(mm)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)']]
y_train = train_df['2이송 인원']

# 테스트 데이터에서 필요한 열 선택
X_test = test_df[['Year', 'Month', 'Day', 'Hour', '3최고기온', '4평균기온', '9평균습도(%)',
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
print("-----KFold 교차검증 결과-----")
print(f'KFold Cross-Validation MSE: {np.mean(mse_list):.2f}')
print(f'KFold Cross-Validation RMSE: {np.mean(rmse_list):.2f}')
print(f'KFold Cross-Validation MAE: {np.mean(mae_list):.2f}')
print(f'KFold Cross-Validation R^2: {np.mean(r2_list):.2f}')

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

print("-----성능평가 결과-----")
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R^2: {r2:.2f}')

# 10. 결과 저장
test_df['patient prediction'] = y_test_pred
test_df.to_csv('./data/6.Heatstroke_test_predictions.csv', index=False)

# 11. 예측 결과 시각화
plt.figure(figsize=(12, 8))
indices = range(len(y_test))  # 인덱스 생성

# 결과값 필터링
filtered_indices = [i for i in indices if 20 <= y_test.values[i] <= 160]
filtered_actual = [y_test.values[i] for i in filtered_indices]
filtered_predicted = [y_test_pred[i] for i in filtered_indices]

# 데이터프레임으로 변환
plot_df = pd.DataFrame({
    'Index': filtered_indices,
    'Actual': filtered_actual,
    'Predicted': filtered_predicted
})

# 실제 값과 예측 값 시각화
sns.lineplot(data=plot_df, x='Index', y='Actual', marker='o', linestyle='-', label='Actual Values')
sns.lineplot(data=plot_df, x='Index', y='Predicted', marker='x', linestyle='--', color='red', label='Predicted Values')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.savefig('./result/Test_Predictions_GradientBoosting.png')
plt.show()
