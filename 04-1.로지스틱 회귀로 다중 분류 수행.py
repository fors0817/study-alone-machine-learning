# p.188
# 로지스틱 회귀는 max_iter 매개변수를 이용해서 반복 횟수를 지정. 
# 로지스틱 회귀는 계수의 제곱을 규제한다(L2규제). 릿지 회귀는 alpha(커지면 규제도 커짐), 로지스틱 회귀는 C(작을수록 규제도 커짐)의 매개변수로 규제한다.

#max_iter=1000, C=20로 규제하는 다중 로지스틱 회귀

import numpy as np
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
#print(lr.score(train_scaled, train_target)) #0.9327731092436975
#print(lr.score(test_scaled, test_target)) #0.925

## 테스트 세트의 5개의 샘플 추출
#print(lr.predict(test_scaled[:5])) #['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']

## 테스트 세트의 5개의 샘플에 대한 예측 확률 추출(소수점은 세자리까지)
proba = lr.predict_proba(test_scaled[:5])
#print(np.round(proba, decimals=3))

## 확률들의 클래스 정보 확인
#print(lr.classes_) #['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

## 소프트맥스를 활용한 확률
##z1~z7까지 구하기
decision = lr.decision_function(test_scaled[:5])
#print(np.round(decision, decimals=2))

##s1~s7까지 구하기
from scipy.special import softmax
proba = softmax(decision, axis=1)
#print(np.round(proba, decimals=3))





