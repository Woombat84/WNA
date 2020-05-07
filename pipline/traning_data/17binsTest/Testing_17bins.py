import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import math

file = "D:\\Programming\\Python\\P6\\cropped_weednumber_sorted\\training_dataColour.xlsx"
df = pd.read_excel(file)

x = [df['arc_tot'],df['skel_tot'],df['bin1'],df['bin2'],df['bin3'],df['bin4'],df['bin5'],df['bin6'],df['bin7'],df['bin8'],df['bin9'],df['bin10'],df['bin11'],df['bin12'],df['bin13'],df['bin14'],df['bin15'],df['bin16'],df['bin17']]
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
error = 0
for i in range(y_pred.shape[0]):
    error += math.sqrt((y_pred[i]-y_test[i])**2)

error = error/len(y_pred)
print("Average error: ", error)

filename = 'WNA_model.sav'
pickle.dump(model, open(filename, 'wb'))