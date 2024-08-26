# 3.최고온도 ------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['3.Highest Temperature (℃)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Highest Temperature vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Highest Temperature (℃)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Highest_Temperature.png'))
plt.show()

#4평균온도-------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['4.Average Temperature (℃)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Average Temperature vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Average Temperature (℃)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Average_Temperature_vs_Heatstroke_Patients_with_Regression.png'))
plt.show()

#9평균습도-------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# # X와 Y 설정
X2 = df['9.Average Humidity (%)']
Y2 = df['2.Total Transported Patients']
#
# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')



# 그래프에 제목 추가
plt.title("Average Humidity vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Average Humidity (%)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Average_Humidity.png'))
plt.show()

#10강수량합계-------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['10.Total Precipitation (mm)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Total Precipitation vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Total Precipitation (mm)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Total_Precipitation.png'))
plt.show()

#18최고 최저 기온차---------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['18.Temperature Difference (High-Low)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Temperature Differencevs vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Temperature Difference (High-Low)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Temperature_Difference.png'))
plt.show()

#19체감온도--------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['19.Apparent Temperature (℃)']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Apparent Temperature vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Apparent Temperature (℃)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Apparen_Temperature.png'))
plt.show()

#20불쾌지수--------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['20.Discomfort Index']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Discomfort Index vs Number of Heatstroke Patients with Regression Line")
plt.xlabel('Discomfort Index')
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Discomfort_Index.png'))
plt.show()

#38전일 이송 환자 수--------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# X와 Y 설정
X2 = df['39.Transported Patients on Previous Day']
Y2 = df['2.Total Transported Patients']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# Seaborn의 regplot을 사용하여 산점도와 선형 회귀선을 함께 그리기
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')

# 그래프에 제목 추가
plt.title("Transported Patients on Previous Day vs Number of Heatstroke Patients with Regression Line")
plt.xlabel('Transported Patients on Previous Day')
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'Transported Patients on Previous Day.png'))
plt.show()

# -------------------------------------------통합그래프------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 파일 읽기
df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
header_english = [
    '1.Date', '2.Total Transported Patients', '3.Highest Temperature (℃)', '4.Average Temperature (℃)', '5.Lowest Temperature (℃)',
    '6.Sunshine Duration (hours)', '7.Average Wind Speed (m/s)', '8.Average Cloudiness (10-minute ratio)', '9.Average Humidity (%)',
    '10.Total Precipitation (mm)', '11.Minimum Relative Humidity (%)', '12.Total Solar Radiation (MJ/m²)', '13.Average Vapor Pressure (hPa)',
    '14.Average Local Pressure (hPa)', '15.Average Sea Level Pressure (hPa)', '16.Maximum Wind Speed (m/s)', '17.Maximum Gust Wind Speed (m/s)',
    '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)', '20.Discomfort Index', '21.Month', '22.Day of the Week',
    '23.Weekend and Holidays', '24.Day_Clear Percentage', '25.Day_Cloudy Percentage', '26.Day_Rain Percentage', '27.Day_Lightning Presence',
    '28.Night_Clear Percentage', '29.Night_Cloudy Percentage', '30.Night_Rain Percentage', '31.Night_Lightning Presence',
    '32.Difference in High Temp from Previous Day', '33.Difference in Average Temp from Previous Day', '34.Difference in Low Temp from Previous Day',
    '35.5-Day Moving Average of High Temp', '36.5-Day Moving Average of Average Temp', '37.5-Day Moving Average of Apparent Temp',
    '38.5-Day Moving Average of Discomfort Index', '39.Transported Patients on Previous Day', '40.5-Day Moving Average of Transported Patients', '41.Year'
]
df.columns = header_english

# 사용할 컬럼(속성) 이름 지정
column_names = ['3.Highest Temperature (℃)', '4.Average Temperature (℃)', '9.Average Humidity (%)',
                '10.Total Precipitation (mm)', '18.Temperature Difference (High-Low)', '19.Apparent Temperature (℃)',
                '20.Discomfort Index', '39.Transported Patients on Previous Day']

# 색상 목록 정의
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']

df[column_names].plot(kind='density', figsize=(12, 10), subplots=True, layout=(3, 3), sharex=False)

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 밀도 그래프 그리기
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

# Plotting 각 subplot에 대해 밀도 그래프 그리기
for i, col in enumerate(column_names):
    row = i // 3
    col_num = i % 3
    df[col].plot(kind='density', ax=axes[row, col_num], title=col, color=colors[i % len(colors)])

# 빈 서브플롯 비활성화
for j in range(i + 1, 9):
    fig.delaxes(axes.flatten()[j])

# 제목을 그래프 아래쪽에 추가
fig.suptitle("Density Plots of Various Features", y=0.02, fontsize=16)

# 그래프 간의 간격 조정
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# 결과를 파일로 저장
plt.savefig(os.path.join(output_dir, 'density_plots.png'))
plt.show()