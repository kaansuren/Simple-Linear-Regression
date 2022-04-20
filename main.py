import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")
print(data)

x = data.iloc[:,:1].values
y = data.iloc[:,1:].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

predicted = lr.predict(x_test)

print(lr.predict([[11]]))


















