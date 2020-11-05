#!/usr/bin/env

import functions.logger as logger
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
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

        # Network parameters
        n_hidden1 = 10
        n_hidden2 = 10
        n_input = 9
        n_output = 1

        # Learning parameters
        learning_constant = 0.2
        number_epochs = 1000
        batch_size = 20

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

        # Load data
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',
                           sep=",",
                           names=["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig",
                                  "breast", "breast-quad", "irradiat"])

        y_data = pd.DataFrame(data['Class'])
        X_data = data.drop('Class', axis=1)

        # Encoding categorical label
        print(f"Encoding of Class: \n{dict(enumerate (y_data['Class'].astype('category').cat.categories))}\n")
        y_data['Class'] = y_data['Class'].astype('category').cat.codes

        # Encoding categorical features
        categorical_columns = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "breast", "breast-quad",
                               "irradiat"]
        for column in categorical_columns:
            print(f"Encoding of {column}: \n{dict(enumerate(X_data[column].astype('category').cat.categories))}\n")
            X_data[column] = X_data[column].astype('category').cat.codes

        # Set label
        label = y_data.values

        # Split data in Training 70%, Test 15% and Validation 15%
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=seed)
        # Convert to arrays
        X_train, X_test, X_val, y_train, y_test, y_val = X_train.values, X_test.values, X_val.values,\
                                                         y_train.values, y_test.values, y_val.values

        X_train_batches = np.split(X_train, 5)
        y_train_batches = np.split(y_train, 5)

        with tf.Session() as sess:

            sess.run(init)

            # Training epoch
            for epoch in range(number_epochs):

                for X_batch, y_batch in zip(X_train_batches, y_train_batches):
                    sess.run(optimizer, feed_dict={X: X_batch, Y: y_batch})
                # Display the epoch
                if epoch % 100 == 0:
                    print("Epoch:", '%d' % (epoch))

            # Test model
            pred = (neural_network)  # Apply softmax to logits
            accuracy = tf.keras.losses.MSE(pred, Y)
            #print("Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
            # tf.keras.evaluate(pred,)
            #print("Prediction:", pred.eval({X: X_train}))
            output = neural_network.eval({X: X_train})
            result = pd.DataFrame()
            result['label'] = list(y_train)
            result['prediction'] = list(pred.eval({X: X_train}))
            result['output'] = list(neural_network.eval({X: X_train}))
            result['accuracy'] = list(accuracy.eval({X: X_train, Y: y_train}))
            #result = pd.DataFrame({"label": y_train,
            #                       "prediction": pred.eval({X: X_train}),
            #                       "output": neural_network.eval({X: X_train}),
            #                       "accuracy": accuracy.eval({X: X_train, Y: y_train})
            #                       })
            plt.plot(y_train, 'ro', output, 'bo')
            plt.ylabel('some numbers')
            plt.show()

            estimated_class = tf.argmax(pred, 1)  # +1e-50-1e-50
            correct_prediction1 = tf.equal(tf.argmax(pred, 1), label)
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

            print(correct_prediction1)
            print(estimated_class)
            print(accuracy1.eval({X: X_data}))

    def multilayer_perceptron(self, input_d):
        # Task of neurons of first hidden layer
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, self.w1), self.b1))
        # Task of neurons of second hidden layer
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        # Task of neurons of output layer
        out_layer = tf.add(tf.matmul(layer_2, self.w3),self.b3)
        return out_layer


if __name__ == "__main__":
    try:
        TensorFlow()
    except Exception as e:
        raise e
