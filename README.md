
## Self Organizing Maps (SOM):
Dataset: http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)

## Description:
Unsupervised deep learning model is going to identify some patters. Here patters mean customers. So its going to do some customers segmentation to identify segment of customers and one of the segments will have customers that are farud.

## Approch:
All these customers are inputs of the neural network. And this inputs are going to mapped to an output space, between input and output space we have neurons. And each neurons are initilized as a vector of weights that is same size as the vector of customer. That is vector of 15 elements(1 customer id + 14 attribute).

For each customer (i.e observation) the output will be the neuron that is closest to the customer. This neuron is called wining node. The wining node is the most similar neuron to the customer. After this we use neighbour function like gaussian neighbour function to update the weights of the neighbours of wining node to move them closer to the point.

We do this for all the customers and repeat is again. Each time output space decreases and losses dimensions. Finally label is reached where output space stops decreasing.

That is the stage where we get out SOM with all the wining nodes. Ourliers will represent farud customers.

So farud is the outlying neurons because out lying neurons are far from majority of the neurons. In order to find outliers we will find mean of Euclidean distance between each neurons and its neighbour. Neighbour is defined manually. So if neurons is far from neighbour then it is outlier

## Map of wining nodes:
![capture](https://user-images.githubusercontent.com/30834801/43048336-9d2f90b8-8e03-11e8-84ce-caabbed578bb.PNG)


## Fraud List
![fraudlist](https://user-images.githubusercontent.com/30834801/43048345-d5be1f1c-8e03-11e8-8ba6-d0575c4b3cf1.PNG)

## Note: If you run this code it may possible that you will get different result. That is because The SOM is not really a foolproof algorithm. With a small dataset, it has high variance problems and it is quite prone to be stochastic.
