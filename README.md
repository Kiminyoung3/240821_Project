<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Heat?%20Hit!&fontSize=90&animation=fadeIn&fontAlignY=38&desc=일일%20열사병%20예측%20프로그램%20%7C%20현대건설%20기술교육원%20Smart%20안전학과&descAlignY=75&descAlign=62"/>
</p>

> 조장 : 김인영 / 조원 : 최미선, 정유진, 이성준

<div style="text-align: left;"> 
    <h2 style="border-bottom: 1px solid #d8dee4; color: #282d33;">  </h2>  
    <div style="font-weight: 700; font-size: 15px; text-align: left; color: #282d33;">  </div> 
    </div>
    



# 프로젝트 명 : Heat ? Hit !

열사병 환자 예측 모델 개발



<div style="text-align: left;"> 
    <h2 style="border-bottom: 1px solid #d8dee4; color: #282d33;">  </h2>  
    <div style="font-weight: 700; font-size: 15px; text-align: left; color: #282d33;">  </div> 
    </div>

## > 프로젝트 개요
이 프로젝트는 건설현장에서 발생하는 열사병 환자 수를 예측하기 위한 머신러닝 모델을 개발하는 것을 목표로 합니다. 최근 여름철의 기온 상승과 작업 강도 증가로 인해 열사병 발생이 급증하고 있으며, 이를 해결하기 위해 기상 데이터와 작업 환경 데이터를 활용한 예측 모델을 구축했습니다.

## > 목표

* 현장 작업자 안전 관리: 예측 기반의 예방 조치 및 실시간 경고 시스템 구축

* 작업 일정 조정: 열사병 위험도가 높은 날씨 조건을 고려한 작업 일정 조정 및 휴식 주기 설정

* 건강 관리 및 정책 결정: 보건 정책 수립 및 예방 교육 자료 제공, 응급 대응 시스템 구축

## > 데이터

[Heatstroke 자료]https://github.com/Kiminyoung3/240821_Project/blob/main/data/6.Heatstroke_train_new_header.csv

### 독립 변수

* 날짜 및 시간 (Year, Month, Day)

* 일 최고 기온 (Max Temperature)

* 일 평균 기온 (Average Temperature)

* 평균 습도 (Average Humidity)

* 강수량 합계 (Total Precipitation)

* 최고-최저 기온차 (Temperature Range)

* 체감온도 (Apparent Temperature)

* 불쾌지수 (Discomfort Index)

* 전날 열사병 이송 환자 수

* 5일간 평균 열사병 이송 환자 수

* 종속 변수

* 열사병 이송 환자 수 (Heatstroke Cases)


# 개발결과
## Gradient Boosting
장점: 비선형 관계를 잘 모델링할 수 있으며, 높은 정확도를 제공하고 과적합을 방지
성능 지표:

#### * MSE: 약 177.74

#### * RMSE: 약 13.22

#### * MAE: 약 7.42

#### * R²: 0.89

## Random Forest
장점: 변수 간 복잡한 상호작용을 잘 포착하고, 데이터의 잡음에 강함
성능 지표:
#### * MSE: 186.28
#### * RMSE: 13.48
#### * MAE: 7.47
#### * R²: 0.88


## > 결과
Gradient Boosting 모델이 Random Forest 모델보다 전체적으로 더 우수한 성능을 보였으며, 예측 값이 실제 값과 더 유사하게 나타났습니다. 성능 비교 결과, Gradient Boosting 모델이 열사병 예측에 더 적합한 것으로 나타났습니다.

사용 방법
환경 설정: 프로젝트를 실행하기 위해 필요한 라이브러리 및 패키지를 설치합니다.
데이터 준비: 기상 데이터와 작업 환경 데이터를 준비합니다.
모델 학습: Gradient Boosting 또는 Random Forest 모델을 사용하여 데이터로 학습합니다.
예측: 학습된 모델을 사용하여 열사병 환자 수를 예측합니다.
평가: 성능 지표를 활용하여 모델의 성능을 평가합니다.
참고 자료
데이터 시각화 및 분석 결과는 본 문서 내의 그림 및 그래프 섹션에서 확인할 수 있습니다.
성능 비교 및 결과 해석에 대한 자세한 내용은 개발 결과 섹션에서 설명합니다.

## > 프로젝트 설치

필수 소프트웨어 및 라이브러리
이 프로젝트를 실행하기 위해서는 Python 3.7 이상과 다음의 라이브러리들이 필요합니다:

* pandas

* scikit-learn

* numpy                    

* seaborn

* matplotlib


## > 프로젝트 사용 방법

여기에 GradientBoosting.py와 RandomForest.py를 비교 분석하고, correlation.py에서 독립 변수 간 상관관계를 분석하는 과정을 설명하는 코드를 정리했습니다. 각 코드가 수행하는 작업을 이해하고 필요에 따라 수정할 수 있습니다.

### 1. GradientBoosting.py

이 스크립트는 GradientBoostingRegressor를 사용하여 열사병 환자 이송 인원 예측 모델을 훈련하고 평가합니다.

##### 주요 단계:

###### 데이터 로드: train_df와 test_df를 CSV 파일에서 로드합니다.

###### 헤더 설정: 데이터 프레임의 컬럼 이름을 설정합니다.

###### 날짜 및 시간 변환: 1날짜 및 시간 열을 연, 월, 일, 시간으로 변환합니다.

###### 데이터 전처리: 모델 학습에 필요한 특성(X)과 목표 변수(y)를 설정합니다.

###### 스케일링: 특성 데이터를 표준화합니다.

###### 모델 훈련 및 검증: KFold 교차 검증을 사용하여 GradientBoostingRegressor 모델을 평가합니다.

###### 전체 데이터로 모델 훈련: 전체 훈련 데이터를 사용하여 최종 모델을 훈련합니다.

###### 예측 및 성능 평가: 테스트 데이터에 대해 예측을 수행하고 성능을 평가합니다.

###### 결과 저장: 예측 결과를 CSV 파일로 저장합니다.

###### 시각화: 예측 결과를 시각화하여 분석합니다.

### 2. RandomForest.py
이 스크립트는 RandomForestRegressor를 사용하여 열사병 환자 이송 인원 예측 모델을 훈련하고 평가합니다.

##### 주요 단계:

###### 데이터 로드: train_df와 test_df를 CSV 파일에서 로드합니다.

###### 헤더 설정: 데이터 프레임의 컬럼 이름을 설정합니다.

###### 날짜 및 시간 변환: 1날짜 및 시간 열을 연, 월, 일, 시간으로 변환합니다.

###### 데이터 전처리: 모델 학습에 필요한 특성(X)과 목표 변수(y)를 설정합니다.

###### 스케일링: 특성 데이터를 표준화합니다.

###### 모델 훈련 및 검증: KFold 교차 검증을 사용하여 RandomForestRegressor 모델을 평가합니다.

###### 전체 데이터로 모델 훈련: 전체 훈련 데이터를 사용하여 최종 모델을 훈련합니다.

###### 예측 및 성능 평가: 테스트 데이터에 대해 예측을 수행하고 성능을 평가합니다.

###### 결과 저장: 예측 결과를 CSV 파일로 저장합니다.

###### 시각화: 예측 결과를 시각화하여 분석합니다.

### 3. correlation.py
이 스크립트는 독립 변수 간의 상관관계를 분석합니다.

##### 주요 단계:

###### 데이터 로드: train_df를 CSV 파일에서 로드합니다.

###### 헤더 설정: 데이터 프레임의 컬럼 이름을 설정합니다.

###### 날짜 및 시간 변환: 1날짜 및 시간 열을 연, 월, 일, 시간으로 변환합니다.

###### 상관관계 분석: 독립 변수 간의 상관관계를 분석합니다.

###### 상관관계 시각화: 상관관계 매트릭스를 시각화하여 변수 간의 관계를 분석합니다.

# >  Gradient Boosting 학습 모델

![Gradient Boosting 회귀 모델](https://github.com/user-attachments/assets/320158af-8438-4c44-9ecf-59abaa8cd1cc)

# > Random Forest 학습 모델

![Random Forest 회귀 모델](https://github.com/user-attachments/assets/d6db0ea1-0f22-4bb2-810a-9c1889b12ea5)

## > 머신러닝 모델의 한계

#### 기후 변화와 갑작스러운 날씨 변동은 예측 변수의 변동성을 증가시킬 수 있다. 

예측 시점에 갑작스러운 기온 상승이나 강수량의 급격한 변화는 모델이 학습한 패턴과 다를 수 있으며, 이로 인해 모델의 예측 정확도에 영향을 줄 수 있다. 기후 변화로 인해 장기적인 기온 상승이 지속되면, 기존 모델이 예측하는 패턴과 실제 패턴 간의 불일치가 발생할 수 있다. 

## > 차후 연구 계획획

#### 기후 변화와 날씨 변동성을 고려하여, 최신 기후 데이터를 반영하고 장기적인 기후 트렌드를 모델에 통합하는 것이 필요하다. 

예측 모델에 최신 기후 데이터와 변동성 요소를 추가하고, 기후 변화에 대한 적응력을 높이기 위해 주기적인 모델 업데이트를 수행하여 불확실성을 줄일 수 있을 것으로 보인다.

![Footer](https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=footer)
