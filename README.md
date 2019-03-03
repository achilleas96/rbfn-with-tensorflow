# **RBF NETWORK with Tensorflow**

The  theory behind this architecture can be foun in the [article](http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial)

**DataSet** 

The dataset used for this network is mnist which has 55000 images (table 55000x784) as training set and 10000 images (table 10000x784) as test set. 
Each image has a size of 784 pixels. The data set is imported from tensorflow (tensorflow.examples.tutorials.mnist)

**Architecture**

  * The network can seperated in two different subnetworks. The first one has as input layer the input vector and output layer the rbf neurons. The second subnetwork has as input layer the rbf neurons and output layer the output layer of thehole network. 

  * **First subnetwork**

    To calculate the center and the radius(σ) of each rbf neuron we use the algorithm kmeans. 

    Input  vector (image): [1,784]

    Output vector : [1,k], where k is the number of rbf neurons



  * **Second subnetwork**

    The second subnetwork is a [multilayer percetron](https://skymind.ai/wiki/multilayer-perceptron) with no hidden layers.
To train the network the tensorflow framework is used

    Input vector: [1,k], the output of the first subnetwork

    Output vector: [1,n], the number of different classes in our case n = 10

**Execution**

To run an experiment using this network the only thing you need to do is set the variable k at method experiment. 
The results are the accuracy of the network on training and test set.

**Results**

Number of rbf neurons: 410

Test accuracy: 87.48%

Train accuracy: 86.56%

![Alt text](https://github.com/achilleas96/rbfn-with-tensorflow/blob/master/k%3D410.png?raw=true "Title")

