###
###
# This is an implementation of the kNN algorithm. The code is written with
# readability in mind, and most importantly, learning how the algorithm works.
# I recommend using kNN from a generic machine learning library (sci-kit learn)
# as those are usually optimized for performance.
###
### By: Zak Ahmad (zakahmad@gatech.edu)
###
# this code will run only if numpy and pandas are installed
# if not, please download numpy from http://www.scipy.org/scipylib/download.html
# and pandas from http://pandas.pydata.org/getpandas.html
import numpy as np
import pandas as pd
import random

# For more information about the 1974 Motor Trend Car Road Test dataset 
# see http://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html

# --------------------   read the data  --------------------
# specify file name containing the data
filename = "cardata.csv"
# create a pandas dataframe containing the data in the csv file
cardata = pd.read_csv(filename)
# find the length (number of rows) of data
data_length = len(cardata)
# we will use only 60% of the data to construct our training set
train_length = int(0.6 * data_length)

# ---------------------   prepare data  --------------------
# generate numpy array from zero to length of data used for indices
indices = np.arange(data_length)
# randomize the order of the indices
random.shuffle(indices)
# create indices for training set from the randomized indices
train_indices = indices[0:train_length]
# output feature we are interested in predicting
yfeature = ["mpg"]
# input feature(s) we will use to construct our prediction model
xfeatures = ["cyl","disp","wt"]
# this creates the training set which constructs our prediction model
X_train_data = cardata.ix[train_indices,xfeatures]
Y_train_data = cardata.ix[train_indices,yfeature]

# ------------------  predict, using data  -----------------
# numpy array containing the features of the vehicle we want to predict
# note the order of the array elements has to be the same as xfeatures
my_vehicle = np.array([4,80,1.5]) # 4 cylinders, 80 cubic inches, 1.500 lb/1000
# k is the number of elements we will be averaging for our prediction
k = 3
if k < 1:
    # k should atleast be 1    
    k = 1
elif k > len(Y_train_data):
    # k cannot be more than the training set's length
    k = len(Y_train_data)
# this is where the "prediction begins", we compute the Euclidean distance
# squared difference between my_vehicle and all x training set (input features)
sqdiff = (X_train_data - my_vehicle)**2
# now compute the Euclidean distance for each row (axis = 1)
dist = np.sqrt(sqdiff.sum(axis=1))
# store the Euclidean distance into a new column in the y training set (output)
Y_train_data["dist"] = dist
# sort the y training set by the Euclidean distance 
sorted_output = Y_train_data.sort("dist")
# get the yfeature as a numpy array and compute mean of only k elements
prediction = np.mean(sorted_output.as_matrix(yfeature)[0:k])
print prediction
