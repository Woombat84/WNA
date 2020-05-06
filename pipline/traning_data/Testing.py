import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import openpyxl
import pathlib

file = "./traning_data.xlsx"
df = pd.read_excel(file)

#print(df)

x = [df['arc_tot'],df['skel_tot'],df['dark_brown'],df['light_green'],df['medium_green'],df['dark_green']]
y = df['weed_number']

#y = y.transpose()
#print(y.shape)
# norm
x, y = np.array(x), np.array(y)

#x = x.reshape(-1, 1)
#x = x.T
#nsamples, nx, ny = x.shape
#x_reshape = train_dataset.reshape((nsamples,nx*ny))

#print(x_reshape.shape)

min = np.amin(x, axis=0)
max = np.amax(x, axis=0)

x_norm = np.zeros(x.shape)

for i in range(x.shape[0]):
    x_norm[i] = (x[i]-min)/(max-min)

#x = x.reshape(-1, 1)

#print(x_norm)
x_norm = x_norm.transpose()
#x_norm = x_norm.reshape(x_norm.shape[1:])
print("Shape of x: ", x_norm.shape)
print("Shape of y: ", y.shape)
X_train, X_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.33, random_state=42)

#print("Shape of X_train: ", X_train.shape)
#print("Shape of y_train: ", y_train.shape)

model = LinearRegression().fit(X_train, y_train)
r_sq = model.score(x_norm, y) #R^2

print("R^2: ",r_sq)

y_pred = model.predict(X_test)

#print("Predicted values: ", y_pred)
#print("Actual values: ", y_test)

filename = 'WNA_model.sav'
pickle.dump(model, open(filename, 'wb'))