# p.183
# 로지스틱 회귀는 이름은 회귀이지만 분류 모델. 시그모이드 함수 또는 로지스틱 함수를 사용해 확률값(0~1)사이값으로 치환. 함수의 모양을 보면 z가 무한한 음수로 가면 0, z가 무한대로 가면 1

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

## 로지스틱 함수를 -5,5까지 0.1 간격으로 그리기
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1/(1+np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
# plt.show()

## 로지스틱 회귀로 이진 분류 수행하기 위해 도미와 빙어를 추출하는 불리언 인덱싱
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

## 로지스틱 회귀 훈련하기
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

## 로지스틱 회귀로 훈련한 5개의 샘플을 뽑아내고 확률들 확인해보기
#print(lr.predict(train_bream_smelt[:5])) ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
#print(lr.predict_proba(train_bream_smelt[:5]))

## 로지스틱 회귀에서 학습계수 확인하기
#print(lr.coef_, lr.intercept_) #[[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]

## z값 출력하기
decisions = lr.decision_function(train_bream_smelt[:5])
#print(decisions) #[-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]

## z값을 시그모이드함수를 거쳐 확률값으로 변환하기
from scipy.special import expit #188
#print(expit(decisions)) #[0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]

## 아직 남음
