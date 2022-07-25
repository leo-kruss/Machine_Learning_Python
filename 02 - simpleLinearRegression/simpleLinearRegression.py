#Simple Linear Regression

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the dataset into the training set and test set
from sklearn.impute import train_test_split



