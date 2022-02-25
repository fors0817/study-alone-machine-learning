# p.199

#SGDClassifier
## 'https://bit.ly/fish_csv_data'를 불러온다.
import numpy as pd
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

## 'https://bit.ly/fish_csv_data'를 불러온다.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

## train_set와 test_set를 나눈다.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

##train_set와 test_set를 표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

## 확률적 경사 하강법 import
from sklearn.linear_model import SGDClassifier

## 손실함수는 log, 에포크는 10로 해서 훈련하고 정확도 점수 출력
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
#print(sc.score(train_scaled, train_target)) #0.773109243697479
#print(sc.score(test_scaled, test_target)) #0.775

## 정확도 점수가 낮다. -> 모델을 이어서 훈련
sc.partial_fit(train_scaled, train_target)
#print(sc.score(train_scaled, train_target)) #0.8151260504201681
#print(sc.score(test_scaled, test_target)) #0.85

## epoch를 계속 늘리면 과대적합이 된다. -> 과대적합이 되기 전 훈련을 종료 
## 훈련을 종료하는 최적의 epoch를 찾기 위해 그래프를 그린다.
import numpy as np
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
#plt.show()

## epoch=100에서 적절함. 반복횟수를 100에 맞추고 다시 훈련
sc = SGDClassifier(loss='log', tol=None, max_iter=100, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) #0.957983193277311
print(sc.score(test_scaled, test_target)) #0.925



