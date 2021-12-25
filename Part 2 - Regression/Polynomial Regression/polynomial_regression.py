#Importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#Visualising the Linear Regression results
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue') 
plt.title('Truth or Bluff (LinearRegression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Linear Regression results
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title('Truth or Bluff (PolynomialRegression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Higher resolution and smoother curve (Polynomial Regression)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (PolynomialRegression) Beautiful version')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

#Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))