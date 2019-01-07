import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)


def next_batch(total_batch, batch_size, iteration):
    next_batch = total_batch[batch_size*iteration:batch_size*iteration+batch_size,:]
    return next_batch


def weight_initializer(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.5))


def bias_initializer(shape):
    return tf.Variable(tf.zeros(shape))


def percetron(x, weight, bias):
    #activation function relu f(x)
    #y=f(xw+b)
    return tf.nn.sigmoid(tf.add(tf.matmul(x, weight), bias))


def model(x, k):

    #VARIABLES

    weights =weight_initializer([k, 10])
    bias = bias_initializer(10)
    #MODEL
    logits = percetron(x, weights, bias)
    return logits


def optimization(y_true, logits, learning_rate):
    #calculates the mean square error
    error =tf.losses.mean_squared_error(y_true, logits)

    #Opitmizer uses  Gradient Desent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(error)
    return train


def rbfn_variables(X_train, k):
    #get the center from KMeans
    kmeans_model = KMeans(n_clusters=k).fit(X_train)
    centroids = np.array(kmeans_model.cluster_centers_)
    # the label for each train dat
    y_labels = kmeans_model.labels_
    std = stds(centroids=centroids, X_train=X_train, y_train=y_labels, k=k)
    return centroids, std


def stds(centroids, X_train, y_train,k):
    train_size = X_train.shape[0]
    std = np.zeros(k)
    class_size = np.zeros(k)
    # Counts the size of each class

    for i in range(0, train_size):
        std[int(y_train[i])] += np.linalg.norm(centroids[int(y_train[i])]-X_train[i])
        class_size[int(y_train[i])] += 1
    # Calculates the average distance of the center to each point
    for i in range(0, k):
        std[i] = std[i] / class_size[i]
    return std


#returns the value of the sample after passing the kernel
def rbf_kernel(centroid, betta, sample):
    #e^(-b||x-μ||)
    exp = -betta*np.power(np.linalg.norm(centroid - sample), 2)
    return np.exp(exp)


def rbfn(centroids, stds, X_train):
    k = len(centroids)
    input_size = X_train.shape[0]
    transformation = np.zeros([input_size, k])
    betta = np.zeros(k)
    #b=1/σ^2
    for i in range(k):
        betta[i] = 1 / (2 * np.power(stds[i], 2))
    #The value of each sample after passing the kernels
    for j in range(0, input_size):
        for i in range(k):
            transformation[j][i] = rbf_kernel(centroid=centroids[i], betta=betta[i], sample=X_train[j])
    return transformation


def experiment(k):

    start_time = time.time()
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels

    #                   RBF START
    centroids, std = rbfn_variables(X_train=X_train, k=k)
    X_train = rbfn(centroids=centroids, stds=std, X_train=X_train)
    X_test = rbfn(centroids=centroids, stds=std, X_train=X_test)
    #                   RBF ENDS

    #                   TENSORFLOW STARTS
    #initialiazation of the variables
    x = tf.placeholder(tf.float32, shape=[None, k])
    y_true = tf.placeholder(tf.float32, [None, 10])
    logits = model(x=x, k=k)
    learning_rate = 0.1
    epochs = 300
    optimizer = optimization(y_true=y_true, logits=logits, learning_rate=learning_rate)
    batch_size = 128
    init = tf.global_variables_initializer()
    total_batch = int(mnist.train.images.shape[0] / batch_size)
    test_accuracy = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            for i in range(total_batch):
                batch_x = next_batch(total_batch=X_train, batch_size=batch_size, iteration=i)
                batch_y = next_batch(total_batch=y_train, batch_size=batch_size, iteration=i)
                sess.run(optimizer, feed_dict={x: batch_x, y_true: batch_y})
            # βρίσκει το  train και test accuracy μετά το τέλος κάθε εποχής εκπαίδευσης
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            train_accuracy[epoch] = sess.run(accuracy, feed_dict={x: X_train, y_true: y_train})
            test_accuracy[epoch] = sess.run(accuracy, feed_dict={x: X_test, y_true: y_test})

    #                           TENSORFLOW ENDS


    #                           RESULTS
    print("Total time of rbfn neurons: {}".format(k))
    print("Test accuracy: {0:.2f}%".format(test_accuracy[epochs-1]*100))
    print("Train accuracy: {0:.2f}%".format(train_accuracy[epochs-1]*100))
    print("Total time: {0:.2f}min".format((time.time()-start_time)/60))
    print()
    print()
    train_error=np.zeros(len(train_accuracy))
    test_error=np.zeros(len(test_accuracy))
    for i in range(len(train_accuracy)):
        train_error[i] = 1-train_accuracy[i]
        test_error[i] = 1-test_accuracy[i]

    plt.suptitle('k=410')
    plt.plot(train_error, color='red')
    plt.plot(test_error, color= 'blue')
    plt.show()

if __name__ == '__main__':
    experiment(k=410)
