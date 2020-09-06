import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

model = LinearRegression()

x = np.array([0,1,2,2.5,3,4,5.5,6,6.5,7]).reshape(-1, 1)
y = np.array([7000,10000,14000,16000,18000,22000,30000,33000,38000,45000])

model.fit(x, y)

ip_list = []
while True:
	ip = input('Enter Point: ').lower()
	if ip == 'exit':
		break
	ip_list.append(float(ip))

X = np.array(ip_list).reshape(-1, 1)

Y = model.predict(X)

plt.plot(X, Y)
plt.show()
