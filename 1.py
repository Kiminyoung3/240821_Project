import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header = ['1날짜 및 시간', '2이송 인원', '3최고기온', '4평균기온', '5최저기온', '6일조시간', '7평균풍속(m/s)', '8평균운량', '9평균습도(%)',
          '10강수량합계(mm)', '11최소상대습도(%)', '12합계 전천 일사량(MJ/m^2)', '13평균 증기압(hPa)', '14평균 현지 기압(hPa)',
          '15평균 해면 기압(hPa)', '16최대풍속(m/s)', '17최대 순간 풍속(m/s)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '21월', '22요일', '23주말 및 공휴일',
          '24낮_맑음 비율', '25낮_흐림 비율', '26낮_비 비율', '27낮_번개 비율', '28밤_맑음 비율', '29밤_흐림 비율', '30밤_비 비율', '31밤_번개 있음',
          '32전일 최고 기온 차이', '33전일 평균 기온 차이', '34전일 최저 기온 차이', '35최고 기온 이동 평균(5일간)', '36평균 기온 이동 평균(5일간)',
          '37체감 온도 이동 평균(5일간)', '38불쾌지수 이동 평균(5일간)', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)', '41Year']
# df.columns = header_english
# header_english = [
#     '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
#     '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
#     '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
#     '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
#     '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
#     '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
#     '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
#     '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
#     '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
#     '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
# ]
df.columns = header

# 3. 변경된 CSV 파일 저장
df.to_csv('./data/6.Heatstroke_train_new_header.csv', index=False)

# 저장된 CSV 파일 다시 읽기
data = pd.read_csv('./data/6.Heatstroke_train_new_header.csv')


# '이송 인원' 열을 기준으로 정렬
sorted_df_by_transport = df.sort_values(by='1날짜 및 시간')

# 정렬된 데이터 출력 (열의 간격이 밀리지 않도록)
with pd.option_context('display.max_columns', None):  # 모든 열을 출력
    print("Sorted by '1날짜 및 시간':")
    print(sorted_df_by_transport.head())

#__________________________23.주말 및 공휴일_________________________________________________________
#
# # X와 Y 설정
# X5 = data['39전일의 이송 인원수']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Discomfort Index vs Number of Heatstroke Patients")
# plt.xlabel("Discomfort Index")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________21.월_________________________________________________________
#
# # X와 Y 설정
# X5 = data['21월']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Discomfort Index vs Number of Heatstroke Patients")
# plt.xlabel("Discomfort Index")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________20.불쾌지수_________________________________________________________
#
# # X와 Y 설정
# X5 = data['20불쾌지수']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Discomfort Index vs Number of Heatstroke Patients")
# plt.xlabel("Discomfort Index")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________19.체감온도_________________________________________________________
#
# # X와 Y 설정
# X5 = data['19체감온도']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Apparent Temperature vs Number of Heatstroke Patients")
# plt.xlabel("Apparent Temperature")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________18.최고-최저 기온차_________________________________________________________

# # X와 Y 설정
# X5 = data['18최고-최저 기온차']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Temperature Difference (High-Low) vs Number of Heatstroke Patients")
# plt.xlabel("Temperature Difference (High-Low)")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()


#__________________________12.합계 전천 일사량(MJ/m^2)_________________________________________________________
#
# # X와 Y 설정
# X5 = data['12합계 전천 일사량(MJ/m^2)']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Total Solar Radiation (MJ/m^2) vs Number of Heatstroke Patients")
# plt.xlabel("Total Solar Radiation (MJ/m^2)")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________11.최소상대습도(%)_________________________________________________________

# # X와 Y 설정
# X5 = data['11최소상대습도(%)']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Average Humidity(%) vs Number of Heatstroke Patients")
# plt.xlabel("Average Humidity(%)")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________10.강수량합계_________________________________________________________
#
# # X와 Y 설정
# X5 = data['10강수량합계(mm)']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Average Humidity(%) vs Number of Heatstroke Patients")
# plt.xlabel("Average Humidity(%)")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________9.평균습도_________________________________________________________

# # X와 Y 설정
# X5 = data['9평균습도(%)']
# Y5 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X5, Y5, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Average Humidity(%) vs Number of Heatstroke Patients")
# plt.xlabel("Average Humidity(%)")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________4.평균기온_________________________________________________________
#
# # X와 Y 설정
# X4 = data['4평균기온']
# Y4 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X4, Y4, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Highest Temperature vs Number of Heatstroke Patients")
# plt.xlabel("Highest Temperature")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________3.최고기온_________________________________________________________

# # X와 Y 설정
# X2 = data['3최고기온']
# Y2 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X2, Y2, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Highest Temperature vs Number of Heatstroke Patients")
# plt.xlabel("Highest Temperature")
# plt.ylabel("Number of Heatstroke Patients")
# plt.legend()
# plt.show()

#__________________________1.날짜_________________________________________________________

# # 날짜 및 시간 데이터를 datetime 형식으로 변환 (명시적으로 형식 지정)
# data['날짜 및 시간'] = pd.to_datetime(data['날짜 및 시간'], format='%m/%d/%Y %I:%M:%S %p')
#
# # X와 Y 설정
# X1 = data['1날짜 및 시간']
# Y1 = data['2이송 인원']
#
# # 데이터 분포 시각화 (전체 데이터)
# plt.figure(figsize=(10, 6))
# plt.scatter(X1, Y1, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
# plt.title("Date vs Number of Heatstroke Patients")
# plt.xlabel("Date(12:00 AM)")
# plt.ylabel("Number of Heatstroke Patients")
# plt.xticks(rotation=45)  # 날짜가 겹치지 않도록 회전
# plt.legend()
# plt.show()
