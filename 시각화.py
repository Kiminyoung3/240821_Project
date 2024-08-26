import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 읽기
train_df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 2. 새로운 헤더 설정
train_header = ['1날짜 및 시간', '2이송 인원', '3최고기온', '4평균기온', '5최저기온', '6일조시간', '7평균풍속(m/s)', '8평균운량', '9평균습도(%)',
                '10강수량합계(mm)', '11최소상대습도(%)', '12합계 전천 일사량(MJ/m^2)', '13평균 증기압(hPa)', '14평균 현지 기압(hPa)',
                '15평균 해면 기압(hPa)', '16최대풍속(m/s)', '17최대 순간 풍속(m/s)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '21월', '22요일',
                '23주말 및 공휴일',
                '24낮_맑음 비율', '25낮_흐림 비율', '26낮_비 비율', '27낮_번개 비율', '28밤_맑음 비율', '29밤_흐림 비율', '30밤_비 비율', '31밤_번개 있음',
                '32전일 최고 기온 차이', '33전일 평균 기온 차이', '34전일 최저 기온 차이', '35최고 기온 이동 평균(5일간)', '36평균 기온 이동 평균(5일간)',
                '37체감 온도 이동 평균(5일간)', '38불쾌지수 이동 평균(5일간)', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)', '41Year']

train_df.columns = train_header

# 3.최고온도
X2 = train_df['3최고기온']
Y2 = train_df['2이송 인원']

# result 디렉토리 생성
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 최고기온과 이송 인원의 관계를 나타내는 산점도와 회귀선
plt.figure(figsize=(10, 6))
sns.regplot(x=X2, y=Y2, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Highest Temperature vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Highest Temperature (℃)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Highest_Temperature.png'))
plt.show()

# 4. 평균온도
X4 = train_df['4평균기온']
Y4 = train_df['2이송 인원']

plt.figure(figsize=(10, 6))
sns.regplot(x=X4, y=Y4, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Average Temperature vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Average Temperature (℃)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Average_Temperature_vs_Heatstroke_Patients_with_Regression.png'))
plt.show()

# 9. 평균습도
X9 = train_df['9평균습도(%)']
Y9 = train_df['2이송 인원']

plt.figure(figsize=(10, 6))
sns.regplot(x=X9, y=Y9, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Average Humidity vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Average Humidity (%)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Average_Humidity.png'))
plt.show()

# 10. 강수량합계
X10 = train_df['10강수량합계(mm)']
Y10 = train_df['2이송 인원']

plt.figure(figsize=(10, 6))
sns.regplot(x=X10, y=Y10, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Total Precipitation vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Total Precipitation (mm)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Total_Precipitation.png'))
plt.show()

# 18. 최고 최저 기온차
X18 = train_df['18최고-최저 기온차']
Y18 = train_df['2이송 인원']

plt.figure(figsize=(10, 6))
sns.regplot(x=X18, y=Y18, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Temperature Difference vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Temperature Difference (High-Low)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Temperature_Difference.png'))
plt.show()

# 19. 체감온도
X19 = train_df['19체감온도']
Y19 = train_df['2이송 인원']

plt.figure(figsize=(10, 6))
sns.regplot(x=X19, y=Y19, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Apparent Temperature vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Apparent Temperature (℃)")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Apparent_Temperature.png'))
plt.show()

# 20. 불쾌지수
X20 = train_df['20불쾌지수']
Y20 = train_df['2이송 인원']

plt.figure(figsize=(10, 6))
sns.regplot(x=X20, y=Y20, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Discomfort Index vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Discomfort Index")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Discomfort_Index.png'))
plt.show()

# 38. 전일 이송 환자 수
X38 = train_df['39전일의 이송 인원수']
Y38 = train_df['2이송 인원']

plt.figure(figsize=(10, 6))
sns.regplot(x=X38, y=Y38, scatter_kws={'color':'b', 's':10, 'alpha':0.5}, line_kws={'color':'r'}, marker='o')
plt.title("Transported Patients on Previous Day vs Number of Heatstroke Patients with Regression Line")
plt.xlabel("Transported Patients on Previous Day")
plt.ylabel("Number of Heatstroke Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Transported_Patients_on_Previous_Day.png'))
plt.show()

# -------------------------------------------통합그래프------------------------------------------------

# 사용할 컬럼(속성) 이름 지정
column_names = ['3최고기온', '4평균기온', '9평균습도(%)',
                '10강수량합계(mm)', '18최고-최저 기온차', '19체감온도',
                '20불쾌지수', '39전일의 이송 인원수']

# 색상 목록 정의
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']

# 밀도 그래프 그리기
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

for i, col in enumerate(column_names):
    row = i // 3
    col_num = i % 3
    train_df[col].plot(kind='density', ax=axes[row, col_num], title=col, color=colors[i % len(colors)])

# 빈 서브플롯 비활성화
for j in range(i + 1, 9):
    fig.delaxes(axes.flatten()[j])

# 제목을 그래프 아래쪽에 추가
fig.suptitle("Density Plots of Various Features", y=0.02, fontsize=16)

# 그래프
