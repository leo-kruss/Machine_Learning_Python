#Multiple Linear Regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('C:/Users/LeonardoKruss/Documents/Code/machine_learning_python/03 - multipleLinearRegression/50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size = 0.2, random_state = 0)

#training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#predicting the test set results