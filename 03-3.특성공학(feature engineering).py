import pandas as pd
import numpy as np
## 판다스로 'https://bit.ly/perch_csv_data' 불러오기
aa = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = aa.to_numpy()

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42) 

## 사이킷런 변환기(다중)를 이용해서 변환(인클루드 바이어스) - input값들을 가공할 수 있게 데이터 관리?
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

## 다중회귀 클래스 임포트하고 모델 훈련후 점수 확인 - 과소적합문제 해결, 여전히 과대적합문제 존재 -> 특성을 추가해서 score올린다.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
#print(lr.score(train_poly, train_target)) 0.9903183436982126
#print(lr.score(test_poly, test_target)) 0.9714559911594202

## 5제곱의 선형모델을 만든다. -> 점수확인 - 테스트점수가 음수가 된다. -> 규제한다.
poly = PolynomialFeatures(include_bias=False, degree=5)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
#print(lr.score(train_poly, train_target)) 0.9999999999997439
#print(lr.score(test_poly, test_target)) -144.40564423498796

## 규제를 통해 정규화시킴. 또, 어떤 클래스를 통해 공정하게 정규화시킨다.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
