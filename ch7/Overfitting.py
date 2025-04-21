import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# k 값 설정: x=30에서 f(30)이 약 100이 되도록
k = 100 / 25333333.33

# 5차 함수 정의
def f(x):
    return k * (x**5 / 5 - 1000 * x**3 / 3 + 90000 * x)

# 함수 전체 곡선을 위한 x, y 값 생성 (연속 곡선)
x_curve = np.linspace(-30, 30, 300)
y_curve = f(x_curve)

# [-30,30] 구간에서 63개의 점을 랜덤으로 추출 (노이즈 없이)
np.random.seed(42)  # 재현성을 위한 시드 설정
x_samples = np.random.uniform(-30, 30, 63)
y_samples = f(x_samples)

# 데이터 셔플링 후, 처음 13개는 train, 나머지 30개는 test로 분할
indices = np.random.permutation(63)
x_samples = x_samples[indices]
y_samples = y_samples[indices]
x_train, y_train = x_samples[:13], y_samples[:13]
x_test, y_test = x_samples[13:], y_samples[13:]

# 훈련 데이터 준비
x_train_2d = x_train.reshape(-1, 1)

# 그래프 전체 크기와 여백 설정
plt.figure(figsize=(15, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 모델 학습 및 그래프 그리기
# Degree 2
model_deg2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
model_deg2.fit(x_train_2d, y_train)
x_grid = np.linspace(-30, 30, 300).reshape(-1, 1)

plt.subplot(2, 4, 1)
plt.plot(x_grid, model_deg2.predict(x_grid), color='red')
plt.scatter(x_train, y_train, color='blue', label='Train Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 2 (Train)')

plt.subplot(2, 4, 5)
plt.plot(x_grid, model_deg2.predict(x_grid), color='red')
plt.scatter(x_test, y_test, color='green', label='Test Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 2 (Test)')

# Degree 3
model_deg3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
model_deg3.fit(x_train_2d, y_train)

plt.subplot(2, 4, 2)
plt.plot(x_grid, model_deg3.predict(x_grid), color='red')
plt.scatter(x_train, y_train, color='blue', label='Train Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 3 (Train)')

plt.subplot(2, 4, 6)
plt.plot(x_grid, model_deg3.predict(x_grid), color='red')
plt.scatter(x_test, y_test, color='green', label='Test Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 3 (Test)')

# Degree 4
model_deg4 = make_pipeline(PolynomialFeatures(4), LinearRegression())
model_deg4.fit(x_train_2d, y_train)

plt.subplot(2, 4, 3)
plt.plot(x_grid, model_deg4.predict(x_grid), color='red')
plt.scatter(x_train, y_train, color='blue', label='Train Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 4 (Train)')

plt.subplot(2, 4, 7)
plt.plot(x_grid, model_deg4.predict(x_grid), color='red')
plt.scatter(x_test, y_test, color='green', label='Test Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 4 (Test)')

# Degree 10
model_deg10 = make_pipeline(PolynomialFeatures(10), LinearRegression())
model_deg10.fit(x_train_2d, y_train)

plt.subplot(2, 4, 4)
plt.plot(x_grid, model_deg10.predict(x_grid), color='red')
plt.scatter(x_train, y_train, color='blue', label='Train Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 10 (Train)')

plt.subplot(2, 4, 8)
plt.plot(x_grid, model_deg10.predict(x_grid), color='red')
plt.scatter(x_test, y_test, color='green', label='Test Data')
plt.xlim(-30, 30)
plt.ylim(-7, 7)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.title('Degree 10 (Test)')

plt.savefig('Overfitting_test.png', bbox_inches='tight', dpi=300)
plt.close()