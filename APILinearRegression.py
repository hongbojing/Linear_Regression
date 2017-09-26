import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score, mean_squared_error)

house = pd.read_csv("Desktop/kc_house_data.csv")
house.head()
house = house.drop(['id','date'],axis=1)
Y = house.price.values
house = house.drop(['price'], axis=1)
X = house.as_matrix()


house_X_train = X[:-2003]
house_X_test =  X[(21613-3-2000):21613]

house_y_train = Y[:-2003]
house_y_test =  Y[(21613-3-2000):21613]

lr = LinearRegression(normalize=True)
lr.fit(house_X_train, house_y_train)
dataPred = lr.predict(house_X_test)


print('Coefficients: \n', lr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(house_y_test, dataPred))
print('Test OLS R-Square is %.2f' % r2_score(house_y_test, dataPred))
