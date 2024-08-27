import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 읽기
train_df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')

# 새로운 헤더 설정
train_header = ['1날짜 및 시간', '2이송 인원', '3최고기온', '4평균기온', '5최저기온', '6일조시간', '7평균풍속(m/s)', '8평균운량', '9평균습도(%)',
                '10강수량합계(mm)', '11최소상대습도(%)', '12합계 전천 일사량(MJ/m^2)', '13평균 증기압(hPa)', '14평균 현지 기압(hPa)',
                '15평균 해면 기압(hPa)', '16최대풍속(m/s)', '17최대 순간 풍속(m/s)', '18최고-최저 기온차', '19체감온도', '20불쾌지수', '21월', '22요일',
                '23주말 및 공휴일',
                '24낮_맑음 비율', '25낮_흐림 비율', '26낮_비 비율', '27낮_번개 비율', '28밤_맑음 비율', '29밤_흐림 비율', '30밤_비 비율', '31밤_번개 있음',
                '32전일 최고 기온 차이', '33전일 평균 기온 차이', '34전일 최저 기온 차이', '35최고 기온 이동 평균(5일간)', '36평균 기온 이동 평균(5일간)',
                '37체감 온도 이동 평균(5일간)', '38불쾌지수 이동 평균(5일간)', '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)', '41Year']

train_df.columns = train_header

# 날짜 및 시간 열 변환
def process_datetime(df):
    df['1날짜 및 시간'] = pd.to_datetime(df['1날짜 및 시간'], format='%m/%d/%Y %I:%M:%S %p')
    df['Year'] = df['1날짜 및 시간'].dt.year
    df['Month'] = df['1날짜 및 시간'].dt.month
    df['Day'] = df['1날짜 및 시간'].dt.day
    df['Hour'] = df['1날짜 및 시간'].dt.hour
    df.drop(columns=['1날짜 및 시간'], inplace=True)  # 원본 열 삭제

process_datetime(train_df)

# 독립변수와 '이송 인원'에 대한 산점도와 회귀선
features = {
    'Year': '연도(Year)',
    'Month': '월(Month)',
    'Day': '일(Day)',
    '3최고기온': '일 최고 기온 (℃)',
    '4평균기온': '일 평균 기온 (℃)',
    '9평균습도(%)': '평균 습도 (%)',
    '10강수량합계(mm)': '강수량 합계 (mm)',
    '18최고-최저 기온차': '최고-최저 기온차',
    '19체감온도': '체감온도 (℃)',
    '20불쾌지수': '불쾌지수',
    '39전일의 이송 인원수': '전날의 열사병 환자 이송 인원수',
    '40이송 인원수 이동 평균(5일간)': '이송 인원수 이동 평균(5일간)'
}

output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

# 산점도와 회귀선 그래프를 3x4 형태로 배치
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()  # 2D 배열을 1D 배열로 변환하여 인덱싱을 용이하게 함

for i, (col, label) in enumerate(features.items()):
    if i < len(axes):
        X = train_df[col]
        Y = train_df['2이송 인원']
        sns.regplot(x=X, y=Y, scatter_kws={'color': 'green', 's': 10, 'alpha': 0.5}, line_kws={'color': 'orange'}, ax=axes[i])
        axes[i].set_title(f"{label} vs 열사병 환자 이송 인원수")
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("열사병 환자 이송 인원수")
        axes[i].grid(True)

# 빈 서브플롯 비활성화
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Heatstroke Patient Transportation Counts vs Features", y=1.02, fontsize=20)
plt.savefig(os.path.join(output_dir, 'combined_scatter_plots.png'))
plt.show()

# 밀도 그래프를 3x4 형태로 배치
column_names = ['Year', 'Month', 'Day', '3최고기온', '4평균기온', '9평균습도(%)',
                '10강수량합계(mm)', '18최고-최저 기온차', '19체감온도', '20불쾌지수',
                '39전일의 이송 인원수', '40이송 인원수 이동 평균(5일간)']

colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'grey', 'pink', 'skyblue']

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()  # 2D 배열을 1D 배열로 변환하여 인덱싱을 용이하게 함

for i, col in enumerate(column_names):
    if col in train_df.columns:
        train_df[col].plot(kind='density', ax=axes[i], title=col, color=colors[i % len(colors)])

# 빈 서브플롯 비활성화
for j in range(len(column_names), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Density Plots of Features", y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'density_plots.png'))
plt.show()
