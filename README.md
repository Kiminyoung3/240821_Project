Heatstroke Patient Prediction Model
이 프로젝트는 열사병 환자 예측 모델을 구축하는 과정을 다룹니다. 이 문서에서는 프로젝트의 목적, 데이터 처리, 모델 훈련 및 평가 방법을 단계별로 설명합니다.

목차
프로젝트 개요
설치 방법
사용 방법
데이터 설명
코드 설명
결과
프로젝트 개요
이 프로젝트는 열사병 환자 수를 예측하기 위해 Gradient Boosting Regressor를 사용하여 예측 모델을 구축하는 것입니다. 데이터 전처리, 모델 훈련, 평가 및 결과 시각화를 포함합니다.

설치 방법
이 프로젝트를 실행하려면 다음의 라이브러리가 필요합니다:

pandas
numpy
matplotlib
seaborn
scikit-learn
필요한 라이브러리를 설치하려면 아래 명령어를 실행하세요:

bash
코드 복사
pip install pandas numpy matplotlib seaborn scikit-learn
사용 방법
데이터 준비: ./data/6.Heatstroke_patient_prediction_train_data.csv와 ./data/6.Heatstroke_patient_prediction_test_data.csv 파일을 준비합니다. 이 파일들은 각각 훈련 데이터와 테스트 데이터를 포함하고 있습니다.

데이터 전처리: 날짜 및 시간 열을 분할하여 연도, 월, 일, 시간 정보를 추출합니다. 이후 필요한 열만 선택하여 모델에 맞게 변환합니다.

모델 훈련 및 평가:

KFold 교차 검증을 사용하여 모델을 훈련하고 평가합니다.
최종 모델을 전체 훈련 데이터로 학습시킨 후 테스트 데이터에 대해 예측을 수행합니다.
결과 시각화: 예측 결과를 시각화하여 모델의 성능을 평가합니다.

데이터 설명
훈련 데이터: ./data/6.Heatstroke_patient_prediction_train_data.csv
테스트 데이터: ./data/6.Heatstroke_patient_prediction_test_data.csv
각 데이터 파일은 다음과 같은 열을 포함합니다:

날짜 및 시간
이송 인원
기온, 습도, 강수량 등 기상 관련 변수
코드 설명
1. 데이터 로드
python
코드 복사
train_df = pd.read_csv('./data/6.Heatstroke_patient_prediction_train_data.csv')
test_df = pd.read_csv('./data/6.Heatstroke_patient_prediction_test_data.csv')
훈련 데이터와 테스트 데이터를 로드합니다.

2. 열 이름 변경
훈련 데이터와 테스트 데이터의 열 이름을 새로 설정하여 일관성을 유지합니다.

3. 날짜 및 시간 처리
'1날짜 및 시간' 열을 연도, 월, 일, 시간으로 변환하여 분석에 적합한 형식으로 변환합니다.

4. 데이터 전처리
훈련 데이터와 테스트 데이터에서 필요한 열만 선택하고, 데이터를 스케일링하여 모델에 입력합니다.

5. KFold 교차 검증
데이터를 5개의 폴드로 나누어 모델을 훈련하고, 교차 검증을 통해 모델의 성능을 평가합니다.

6. 모델 훈련 및 예측
Gradient Boosting Regressor를 사용하여 전체 훈련 데이터로 모델을 학습시키고, 테스트 데이터에 대해 예측을 수행합니다.

7. 성능 평가
예측 결과의 MSE, RMSE, MAE, R^2 점수를 계산하여 모델의 성능을 평가합니다.

8. 결과 저장 및 시각화
예측 결과를 CSV 파일로 저장하고, 예측 결과를 시각화하여 모델의 성능을 확인합니다.

결과
최종적으로 생성된 결과 파일:

./result/6.Heatstroke_test_predictions_GradientBoosting.csv: 테스트 데이터에 대한 예측 결과
./result/GradientBoosting_result.png: 예측 결과 시각화
