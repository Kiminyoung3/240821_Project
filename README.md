 
<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Heat?%20Hit!&fontSize=90&animation=fadeIn&fontAlignY=38&desc=일일%20열사병%20예측%20프로그램%20&descAlignY=51&descAlign=62"/>
</p>

프로젝트 명 : Heat ? Hit !

프로젝트 설명: 열사병 환자 예측 모델 개발

○ 프로젝트 개요
이 프로젝트의 목표는 열사병 환자 수를 예측하는 모델을 개발하는 것입니다. 이를 위해 두 가지 회귀 모델, 즉 Gradient Boosting Regressor와 Random Forest Regressor를 비교하고 성능을 평가합니다. 모델 학습 및 평가에는 다양한 기상 데이터와 환자 이송 기록을 사용하며, 결과를 시각화하여 모델의 예측 성능을 분석합니다.



○ 프로젝트 설명

GradientBoosting.py와 RandomForest.py 스크립트는 두 가지 다른 회귀 모델(Gradient Boosting Regressor와 Random Forest Regressor)을 사용하여 열사병 환자 수 예측 문제를 다룹니다. correlation.py는 데이터 전처리 및 상관관계 분석을 위한 스크립트입니다. 여기서는 주요 코드와 기능을 비교하고 설명하겠습니다.

주요 기능 비교
1. 데이터 로드 및 헤더 설정:

두 스크립트 모두 훈련 및 테스트 데이터를 로드하고, 데이터의 열 이름을 새로 설정합니다.


2. 날짜 및 시간 처리:

날짜 및 시간을 연, 월, 일, 시간으로 분할합니다. 두 스크립트에서 이 작업을 동일하게 처리합니다.


3. 데이터 전처리:

훈련 및 테스트 데이터를 선택하고 스케일링을 진행합니다. 스케일링은 두 스크립트 모두 StandardScaler를 사용하여 수행합니다.


4. 모델 훈련 및 평가:

GradientBoosting.py: GradientBoostingRegressor를 사용하여 KFold 교차 검증을 수행하고, 최종 모델을 전체 훈련 데이터로 학습시킨 후 테스트 데이터로 예측을 수행합니다.
RandomForest.py: RandomForestRegressor를 사용하여 KFold 교차 검증을 수행하고, 최종 모델을 전체 훈련 데이터로 학습시킨 후 테스트 데이터로 예측을 수행합니다.
성능 평가 및 결과 저장:

두 스크립트 모두 MSE, RMSE, MAE, R^2를 계산하여 성능을 평가하고 결과를 CSV 파일로 저장합니다.
예측 결과 시각화:

예측 결과를 시각화하여 실제 값과 예측 값의 산점도, 히스토그램, 선 그래프를 생성합니다.

○ 프로젝트 설치 및 실행 방법

필수 소프트웨어 및 라이브러리
이 프로젝트를 실행하기 위해서는 Python 3.7 이상과 다음의 라이브러리들이 필요합니다:

■ pandas

■ numpy

■ matplotlib

■ seaborn

■ scikit-learn

○ 프로젝트 사용 방법

여기에 GradientBoosting.py와 RandomForest.py를 비교 분석하고, correlation.py에서 독립 변수 간 상관관계를 분석하는 과정을 설명하는 코드를 정리했습니다. 각 코드가 수행하는 작업을 이해하고 필요에 따라 수정할 수 있습니다.

1. GradientBoosting.py

이 스크립트는 GradientBoostingRegressor를 사용하여 열사병 환자 이송 인원 예측 모델을 훈련하고 평가합니다.

주요 단계:
데이터 로드: train_df와 test_df를 CSV 파일에서 로드합니다.
헤더 설정: 데이터 프레임의 컬럼 이름을 설정합니다.
날짜 및 시간 변환: 1날짜 및 시간 열을 연, 월, 일, 시간으로 변환합니다.
데이터 전처리: 모델 학습에 필요한 특성(X)과 목표 변수(y)를 설정합니다.
스케일링: 특성 데이터를 표준화합니다.
모델 훈련 및 검증: KFold 교차 검증을 사용하여 GradientBoostingRegressor 모델을 평가합니다.
전체 데이터로 모델 훈련: 전체 훈련 데이터를 사용하여 최종 모델을 훈련합니다.
예측 및 성능 평가: 테스트 데이터에 대해 예측을 수행하고 성능을 평가합니다.
결과 저장: 예측 결과를 CSV 파일로 저장합니다.
시각화: 예측 결과를 시각화하여 분석합니다.

2. RandomForest.py
이 스크립트는 RandomForestRegressor를 사용하여 열사병 환자 이송 인원 예측 모델을 훈련하고 평가합니다.

주요 단계:
데이터 로드: train_df와 test_df를 CSV 파일에서 로드합니다.
헤더 설정: 데이터 프레임의 컬럼 이름을 설정합니다.
날짜 및 시간 변환: 1날짜 및 시간 열을 연, 월, 일, 시간으로 변환합니다.
데이터 전처리: 모델 학습에 필요한 특성(X)과 목표 변수(y)를 설정합니다.
스케일링: 특성 데이터를 표준화합니다.
모델 훈련 및 검증: KFold 교차 검증을 사용하여 RandomForestRegressor 모델을 평가합니다.
전체 데이터로 모델 훈련: 전체 훈련 데이터를 사용하여 최종 모델을 훈련합니다.
예측 및 성능 평가: 테스트 데이터에 대해 예측을 수행하고 성능을 평가합니다.
결과 저장: 예측 결과를 CSV 파일로 저장합니다.
시각화: 예측 결과를 시각화하여 분석합니다.

3. correlation.py
이 스크립트는 독립 변수 간의 상관관계를 분석합니다.

주요 단계:
데이터 로드: train_df를 CSV 파일에서 로드합니다.
헤더 설정: 데이터 프레임의 컬럼 이름을 설정합니다.
날짜 및 시간 변환: 1날짜 및 시간 열을 연, 월, 일, 시간으로 변환합니다.
상관관계 분석: 독립 변수 간의 상관관계를 분석합니다.
상관관계 시각화: 상관관계 매트릭스를 시각화하여 변수 간의 관계를 분석합니다.


○ Heat ? Hit ! 프로젝트 참여자

조장 : 김인영
부원 : 최미선, 정유진, 이성준
