# Self Organizing Maps(SOM)
# Here problem statement is to detect farud customers.
# Dataset: http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)  

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pylab import bone, pcolor, colorbar, plot, show

dataset = pd.read_csv("C:\\Users\\Chandan.S\\Desktop\\DeepLearning\\SelfOrganizingMaps(SOM)\\Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Train SOM
# sigma =1.0 is radius of neighbour
from minisom import MiniSom
som = MiniSom(x=10,y=10, input_len=15, sigma=1.0, learning_rate=0.5)

# Initialize weight randomly
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)


#initilize the window that will contain the map
bone()

# Put all wining node on the map. For that we are going to add mean inter neuron distance on the map for all the wining nodes.
# Different colors mean different range of mean inter neuron distance.
# Get mean distance using distance_map() : this method will return all mean distance in matrix.
# Fraud are decided based on outlier wining nodes shown is white color having maximum mean distance.
pcolor(som.distance_map().T)
colorbar()

# Adding markers
# Red circle : Customers who did not get the approval
# Green Square : Customers who got approval.
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markeredgecolor = colors[y[i]],
         markerfacecolor = 'None', markersize = 10, markeredgewidth =2)
show()  
 
# Finding list of farud customers
# win_map() this method will return dictionory of all mapping from wining nodes to customers

mapping = som.win_map(X)

# Fraud list
# Find co-ordinate of outlier wining node. That is white color wining node co-ordinate(8,1) and (6,8)
fraud = np.concatenate((mapping[(8,1)], mapping[(6,8)]), axis =0)
fraud = sc.inverse_transform(fraud)