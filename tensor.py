import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py


def load_dataset():
    data = np.genfromtxt("Iris Data.txt", delimiter=',')
    iris_features = np.concatenate ((np.concatenate ((data[0:30 , :4] , data[50:80 , :4])) , data[100:130 , :4]))
    iris_features_test = np.concatenate ((np.concatenate ((data[30:50 , :4] , data[80:100 , :4])) , data[130:150 , :4]))
    iris_labels = np.zeros((90, 3))
    iris_labels_test = np.zeros((60, 3))

    for i in range(90):
        if i < 30:
            iris_labels[i] = [1, 0, 0]
        if i >= 30:
            iris_labels[i] = [0 , 1 , 0]
        if i > 59:
            iris_labels[i] = [0 , 0 , 1]

    for i in range (60):
        if i < 20:
            iris_labels_test[i] = [1 , 0 , 0]
        if i >= 20:
            iris_labels_test[i] = [0 , 1 , 0]
        if i > 39:
            iris_labels_test[i] = [0 , 0 , 1]

    return iris_features, iris_labels, iris_features_test, iris_labels_test


def initialize_parameters(check, layers_dims):
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = tf.get_variable("W"+str(l), [layers_dims[l], layers_dims[l-1]],
                                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b"+str(l)] = check * tf.get_variable("b"+str(l), [layers_dims[l], 1],
                                                         initializer=tf.contrib.layers.xavier_initializer (seed=1))

    return parameters


def forward_prop(X, parameters, layers):
    A = X
    for i in range(layers):
        Z = tf.add(tf.matmul(parameters["W"+str(i+1)], A), parameters["b"+str(i+1)])
        A = tf.nn.relu(Z)
    return Z


def cost(ZL, Y):
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def tensor_model(X_train, Y_train, X_test, Y_test,layers, learning_rate, epochs, check_bias, print_cost=True):
    ops.reset_default_graph ()
    tf.set_random_seed(0)
    costs = []
    X = tf.placeholder (tf.float32, shape=[4 , None] , name="Placeholder")
    Y = tf.placeholder (tf.float32, shape=[3 , None] , name="Placeholder")
    layers_dims = layers
    parameters = initialize_parameters(check_bias,layers_dims=layers_dims)
    ZL = forward_prop(X, parameters, len(layers_dims)-1)
    _cost = cost(ZL, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(_cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            _, loss = sess.run([optimizer, _cost], feed_dict={X: X_train.T, Y: Y_train.T})
            if print_cost == True and i % 100 == 0:
                print ("Cost after epoch %i: %f" % (i , loss))
            if i % 5 == 0:
                costs.append(loss)

        plt.plot (np.squeeze (costs))
        plt.ylabel ('cost')
        plt.xlabel ('iterations (per tens)')
        plt.title ("Learning rate =" + str (learning_rate))
        plt.show ()

        parameters = sess.run (parameters)
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
        accuracy = tf.reduce_mean (tf.cast (correct_prediction , "float"))

        print ("Train Accuracy:" , accuracy.eval ({X: X_train.T , Y: Y_train.T}))
        print ("Test Accuracy:" , accuracy.eval ({X: X_test.T , Y: Y_test.T}))

    return parameters


if __name__ == '__main__':
    X_trian , Y_trian , X_test , Y_test = load_dataset ()
    parameters = tensor_model(X_trian , Y_trian , X_test , Y_test, layers=[4 , 10, 5, 5, 3], learning_rate=.035,
                              epochs=2500, check_bias=0, print_cost=False)
