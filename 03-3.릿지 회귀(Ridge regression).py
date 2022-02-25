# p.160

from turtle import color
import pandas as pd
import numpy as np
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

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False, degree=5)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

## 위에 코딩은 특성공학 코딩과 같음
## 릿지 모듈을 호출하고 점수를 확인. -> 훈련 세트에 과대적합되지 않음
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
#print(ridge.score(train_scaled, train_target)) 0.9896101671037343
#print(ridge.score(test_scaled, test_target)) 0.9790693977615381 

## score()메서드의 결과를 저장할 리스트 생성
import matplotlib.pyplot as plt
train_score = []
test_score = []

## alpha값의 범위는 0.001~100까지 10배씩 늘린다.
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ## 릿지 모델을 만든다.
    ridge = Ridge(alpha = alpha)
    ## 릿지 모델을 훈련한다.
    ridge.fit(train_scaled, train_target)
    ## 훈련 점수와 테스트 점수를 저장한다.
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

## 그래프를 그린다.(로그함수) -> 그래프 모양을 보고 적절한 alpha값을 구한다. -> alpha = 0.1
plt.plot(np.log10(alpha_list), train_score, color='red')
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
#plt.show()

## 적절한 alpha값으로 최종 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
#print(ridge.score(train_scaled, train_target)) 0.9903815817570366
#print(ridge.score(test_scaled, test_target)) 0.9827976465386884
 

