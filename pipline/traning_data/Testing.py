import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import openpyxl
import pathlib
import math

file = "./traning_data.xlsx"
df = pd.read_excel(file)

x = [df['arc_tot'],df['skel_tot'],df['ligth_brown'],df['dark_brown'],df['light_green'],df['medium_green'],df['medium_dark_green'],df['dark_green']]

y = df['weed_number']

x, y = np.array(x), np.array(y)

min = np.amin(x, axis=0)
max = np.amax(x, axis=0)


x_norm = np.zeros(x.shape)

for i in range(x.shape[0]):
    x_norm[i] = (x[i]-min)/(max-min)

x_norm = x_norm.transpose()

X_train, X_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.33, random_state=42)

model = LinearRegression().fit(X_train, y_train)
r_sq = model.score(x_norm, y) #R^2

print("R^2: ",r_sq)

y_pred = model.predict(X_test)

#print("Predicted values: ", y_pred)
#print("Actual values: ", y_test)

error = 0
for i in range(y_pred.shape[0]):
    #print("Error: ", y_pred[i]-y_test[i])
    error += math.sqrt((y_pred[i]-y_test[i])**2)

error = error/len(y_pred)
print("Average error: ", error)


filename = 'WNA_model.sav'
pickle.dump(model, open(filename, 'wb'))