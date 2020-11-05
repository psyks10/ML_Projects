#!/usr/bin/env

import functions.logger as logger
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class TensorFlow():

    def __init__(self):

        # Set seed
        seed = 101
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # Initialise logger
        self.logger = logger.initialise_logger()

        # Load data
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',
                           sep=",",
                           names=["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig",
                                  "breast", "breast-quad", "irradiat"])


        # Network parameters
        n_hidden1 = 10
        n_hidden2 = 10
        n_input = 2
        n_output = 2

        # Learning parameters
        learning_constant = 0.2
        number_epochs = 1000
        batch_size = 1000

        # Defining the input and the output
        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_output])

        # DEFINING WEIGHTS AND BIASES

        # Biases first hidden layer
        self.b1 = tf.Variable(tf.random_normal([n_hidden1]))
        # Biases second hidden layer
        self.b2 = tf.Variable(tf.random_normal([n_hidden2]))
        # Biases output layer
        self.b3 = tf.Variable(tf.random_normal([n_output]))

        # Weights connecting input layer with first hidden layer
        self.w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
        # Weights connecting first hidden layer with second hidden layer
        self.w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
        # Weights connecting second hidden layer with output layer
        self.w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

        # Create model
        neural_network = self.multilayer_perceptron(X)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
        # loss_op =
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

        # Initializing the variables
        init = tf.global_variables_initializer()

        y = pd.DataFrame(data['Class'])
        X = data.drop('Class', axis=1)

        label = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)  # Test is 30%
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=seed)  # # 0.5 x 0.3 = 0.15
        X_train, X_test, X_val, y_train, y_test, y_val = X_train.transpose().to_numpy(), X_test.transpose().to_numpy(), X_val.transpose().to_numpy(),\
                                                         y_train.transpose().to_numpy(), y_test.transpose().to_numpy(), y_val.transpose().to_numpy()

        print(type(X_train))
        print(X_train[1])
        '''
        train, test = train_test_split(data, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)
        print(len(train), 'train examples')
        print(len(val), 'validation examples')
        print(len(test), 'test examples')

        batch_size = 5  # A small batch sized is used for demonstration purposes
        train = self.df_to_dataset(train, batch_size=batch_size, label='Class')
        test = self.df_to_dataset(test, shuffle=False, batch_size=batch_size, label='Class')
        val = self.df_to_dataset(val, shuffle=False, batch_size=batch_size, label='Class')
        '''



        with tf.Session() as sess:

            sess.run(init)

            # Training epoch
            for epoch in range(number_epochs):
                sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
                # Display the epoch
                if epoch % 100 == 0:
                    print("Epoch:", '%d' % (epoch))

            # Test model
            pred = (neural_network)  # Apply softmax to logits
            accuracy = tf.keras.losses.MSE(pred, Y)
            print("Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
            # tf.keras.evaluate(pred,)
            print("Prediction:", pred.eval({X: X_train}))
            output = neural_network.eval({X: X_train})
            plt.plot(y_train[0:10], 'ro', output[0:10], 'bo')
            plt.ylabel('some numbers')
            plt.show()

            estimated_class = tf.argmax(pred, 1)  # +1e-50-1e-50
            correct_prediction1 = tf.equal(tf.argmax(pred, 1), label)
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

            print(correct_prediction1)
            print(estimated_class)
            print(accuracy1.eval({X: X}))

    def multilayer_perceptron(self, input_d):
        # Task of neurons of first hidden layer
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, self.w1), self.b1))
        # Task of neurons of second hidden layer
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        # Task of neurons of output layer
        out_layer = tf.add(tf.matmul(layer_2, self.w3),self.b3)
        return out_layer

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32, label='target'):
        dataframe = dataframe.copy()
        labels = dataframe.pop(label)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds


if __name__ == "__main__":
    try:
        TensorFlow()
    except Exception as e:
        raise e
