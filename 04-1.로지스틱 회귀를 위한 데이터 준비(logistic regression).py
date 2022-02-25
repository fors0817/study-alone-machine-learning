#p.176

import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

## fish파일의 첫 5개의 행 출력
#print(fish.head())

## Species의 열의 고유값들 추출
#print(pd.unique(fish['Species']))

## Species열을 제외한 나머니 열은 입력 데이터로 변환 -> 데이터프레임에서 5개의 열을 리스트로 변환
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

## fish_input에 5개의 특성이 잘 저장되어 있는지 5개 열 추출
#print(fish_input[:5])

## fish_target 추출 
fish_target = fish['Species'].to_numpy()

## 데이터 세트 2개 필요
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

## 데이터 세트에 대해 표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

## k-최근접 이웃 분류기의 확률 예측 + 점수 확인
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
#print(kn.score(train_scaled, train_target))
#print(kn.score(test_scaled, test_target))

## k-최근접 이웃 분류기에서 정렬된 출력값 
#print(kn.classes_) ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

## k-최근접 이웃 분류기에서 예측된 5개의 출력값
#print(kn.predict(test_scaled[:5]))

## k-최근접 이웃 분류기에서 예측된 5개의 출력값의 확률(round, decimals이용) -> 출력순서는 classes와 같다.
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
#print(np.round(proba, decimals=4))

## proba의 네번째 샘플의 확률을 확인(distances, indexes 이용) -> 위의(k-최근접 이웃 분류기에서 예측된 5개의 출력값의 확률)과 같음
distances, indexes = kn.kneighbors(test_scaled[3:4])
#print(train_target[indexes])[['Roach' 'Perch' 'Perch']]