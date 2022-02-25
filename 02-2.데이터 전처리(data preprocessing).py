# 도미와 빙어 데이터 
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 도미와 빙어 데이터 합치기
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

# 넘파이 불러오기
import numpy as np

# fish_length와 fish_weight 합치기
fish_data = np.column_stack((fish_length, fish_weight))

# fish_target tuple로 만들기
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# train_test_split()함수를 sklearn의 model_selction 모듈로 불러오기
from sklearn.model_selection import train_test_split

# train_test_splict를 이용해 데이터 무작위로 섞고 random_state 매개변수를 이용해 랜덤 시드 설정
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

# stratify 매개변수를 이용해 클래스 비율에 맞게 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify = fish_target, random_state=42)

#print(test_target)

# K-최근접 이웃 훈련
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# Kneighbors 매서드를 이용해 이웃까지의 거리 확인하기, 기본값이 5이므로 5개 나옴
distances, indexes = kn.kneighbors([[25, 150]])

#평균과 표준편차 구하는 numpy
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)
#print(mean, std)

# 원본 데이터에서 평균을 빼고 표준편차로 나누어 표준점수 변환 -> 모든 행에 대해서 위와 작업을 한 것을 broadcasting이라고 부름
train_scaled = (train_input - mean ) / std

# train_scaled 된 데이터로 산점도(그래프)그려보기
# 샘플[25, 150]도 동일한 비율로 변환
import matplotlib.pyplot as plt
new = ([25, 150]- mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
#plt.show()

kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean ) / std
kn.score(test_scaled, test_target)
#print(kn.predict([new]))

# Kneighbors() 함수로 샘플의 K-최근접 이웃을 구한 다음 산점도로 그림
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()