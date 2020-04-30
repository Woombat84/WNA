import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

file = "./slr01.xls"
df = pd.read_excel(file)

print(df)

x = df['X']
y = df['Y']
#x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
#y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

x = x.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = LinearRegression().fit(X_train, y_train)
r_sq = model.score(x, y) #R^2

print("R^2: ",r_sq)

y_pred = model.predict(X_test)

print("Predicted values: ", y_pred)
print("Actual values: ", y_test)

# Plot outputs
plt.plot(X_test, y_pred, color='black', linewidth=3)
plt.scatter(X_train[:,0], y_train, color="green")
plt.scatter(X_test[:,0], y_test, color='red')


plt.xticks(())
plt.yticks(())
plt.legend(('Regression', 'Training Data', 'Test Data'))

plt.show()