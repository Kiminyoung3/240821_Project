import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_test_data.csv')

# 2. 새로운 헤더 설정
header = ['날짜 및 시간', '이송 인원', '최고기온', '평균기온', '최저기온', '일조시간', '평균풍속(m/s)', '평균운량', '평균습도(%)',
          '강수량합계(mm)', '최소상대습도(%)', '합계 전천 일사량(MJ/m^2)', '평균 증기압(hPa)', '평균 현지 기압(hPa)',
          '평균 해면 기압(hPa)', '최대풍속(m/s)', '최고-최저 기온차', '체감온도', '불쾌지수', '월', '요일', '주말 및 공휴일',
          '낮_맑음 비율', '낮_흐림 비율', '낮_비 비율', '밤_맑음 비율', '밤_흐림 비율', '밤_비 비율', '밤_번개 있음',
          '전일 최고 기온 차이', '전일 평균 기온 차이', '전일 최저 기온 차이', '최고 기온 이동 평균(5일간)', '평균 기온 이동 평균(5일간)',
          '체감 온도 이동 평균(5일간)', '불쾌지수 이동 평균(5일간)', '전일의 이송 인원수', '이송 인원수 이동 평균(5일간)', '연도']
df.columns = header

# 3. 변경된 CSV 파일 저장
df.to_csv('./data/6.Heatstroke_test_new_header.csv', index=False)

# 저장된 CSV 파일 다시 읽기
data = pd.read_csv('./data/6.Heatstroke_test_new_header.csv')

# 날짜 및 시간 데이터를 datetime 형식으로 변환 (명시적으로 형식 지정)
data['날짜 및 시간'] = pd.to_datetime(data['날짜 및 시간'], format='%m/%d/%Y %I:%M:%S %p')

# X와 Y 설정
X1 = data['날짜 및 시간']
Y1 = data['이송 인원']

# 데이터 분포 시각화 (전체 데이터)
plt.figure(figsize=(10, 6))
plt.scatter(X1, Y1, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
plt.title("Datetime vs Number of Heatstroke Patients")
plt.xlabel("Datetime")
plt.ylabel("Number of Heatstroke Patients")
plt.xticks(rotation=45)  # 날짜가 겹치지 않도록 회전
plt.legend()
plt.show()
