#!/usr/bin/env

import functions.logger as logger
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold


class TensorFlow():

    def __init__(self):

        # Set seed
        self.seed = 101
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Initialise logger
        self.logger = logger.initialise_logger()

        # Network parameters
        n_hidden1 = 10
        n_hidden2 = 10
        n_input = 9
        n_output = 2

        # Learning parameters
        learning_constant = 0.02
        number_epochs = 2000
        batch_size = 50

        # Defining the input and the output
        X = tf.placeholder("float", [None, n_input], name="X")
        Y = tf.placeholder("float", [None, n_output], name="Y")

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
        # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Load the data
        X_data, y_data, X_train, X_test, X_val, y_train, y_test, y_val, label = self.load_breast_cancer_data()

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(number_epochs):

                avg_cost = 0.0
                size = int(len(X_train) / batch_size)
                X_train_batches = np.array_split(X_train, size)
                y_train_batches = np.array_split(y_train, size)

                # Mini-batch training
                for X_batch, y_batch in zip(X_train_batches, y_train_batches):
                    _, c = sess.run([optimizer, loss_op], feed_dict={X: X_batch, Y: y_batch})

                # Train all data at once
                # sess.run(optimizer, feed_dict={X: X_train, Y: y_train})

                # Display the epoch
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch}, cost: {c}")

            # Model evaluation

            pred = (neural_network)  # Apply softmax to logits
            accuracy = tf.keras.losses.MSE(pred, Y)

            # Training results
            result_train = self.generate_summary(X_placeholder=X, Y_placeholder=Y,
                                                 X_values=X_train, y_values=y_train,
                                                 accuracy=accuracy, neural_network=pred, session=sess)

            # Accuracy - whole dataset
            result_all = self.generate_summary(X_placeholder=X, Y_placeholder=Y,
                                               X_values=X_data, y_values=y_data,
                                               accuracy=accuracy, neural_network=pred, session=sess)
            estimated_class = tf.argmax(pred, axis=1)  # +1e-50-1e-50
            correct_prediction1 = tf.equal(tf.argmax(pred, axis=1), label)
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

            print(f'Class estimation: {sess.run(estimated_class, feed_dict={X: X_data})}')
            print(f'Correct predictions: {sess.run(correct_prediction1, feed_dict={X: X_data}).sum()} '
                  f'out of {len(label)}')
            print(f'Accuracy: {accuracy1.eval({X: X_data})}')

            # Accuracy - test data
            result_test = self.generate_summary(X_placeholder=X, Y_placeholder=Y,
                                                X_values=X_test, y_values=y_test,
                                                accuracy=accuracy, neural_network=pred, session=sess)

            estimated_class = tf.argmax(pred, axis=1)  # +1e-50-1e-50
            correct_prediction1 = tf.equal(tf.argmax(pred, axis=1), pd.DataFrame(y_test).idxmax(axis=1).values)
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

            print(f'Class estimation: {sess.run(estimated_class, feed_dict={X: X_test})}')
            print(f'Correct predictions: {sess.run(correct_prediction1, feed_dict={X: X_test}).sum()} '
                  f'out of {len(X_test)}')
            print(f'Accuracy: {accuracy1.eval({X: X_test})}')

            # Plot
            output = sess.run(estimated_class, feed_dict={X: X_data})
            plt.plot(y_data, 'ro', output, 'b.')
            plt.ylabel('Class')
            plt.show()

    def generate_summary(self, X_placeholder, Y_placeholder, X_values, y_values, accuracy, neural_network, session):
        result = pd.DataFrame()
        result['label'] = pd.DataFrame(y_values).idxmax(axis=1).values
        result['label_encoded'] = list(y_values)
        result['output'] = list(neural_network.eval({X_placeholder: X_values}))
        result['prediction'] = list(session.run(tf.argmax(neural_network, axis=1), feed_dict={X_placeholder: X_values}))
        result['accuracy'] = list(accuracy.eval({X_placeholder: X_values, Y_placeholder: y_values}))
        return result

    def load_breast_cancer_data(self):
        # Load data
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',
                           sep=",",
                           names=["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig",
                                  "breast", "breast-quad", "irradiat"])

        # Dropping incomplete data - "?"
        data = data.replace('?', np.nan).dropna(axis=0, how='any')
        # Shuffling dataframe
        data = data.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        # Separate features and label
        y_data = pd.DataFrame(data['Class'])
        X_data = data.drop('Class', axis=1)
        # Set label
        label = data['Class'].astype('category').cat.codes
        label = label.values
        # Encode the label using dummies
        y_data = pd.get_dummies(y_data)

        # Encoding categorical features
        categorical_columns = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "breast", "breast-quad",
                               "irradiat"]
        for column in categorical_columns:
            print(f"Encoding of {column}: \n{dict(enumerate(X_data[column].astype('category').cat.categories))}\n")
            X_data[column] = X_data[column].astype('category').cat.codes

        # Split data in Training 60%, Test 20% and Validation 20%
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.40, random_state=self.seed)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=self.seed)
        # Convert to arrays
        X_train, X_test, X_val, y_train, y_test, y_val = X_train.values, X_test.values, X_val.values, \
                                                         y_train.values, y_test.values, y_val.values
        return X_data, y_data, X_train, X_test, X_val, y_train, y_test, y_val, label


    def multilayer_perceptron(self, input_d):
        # Task of neurons of first hidden layer
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, self.w1), self.b1))
        # Task of neurons of second hidden layer
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        # Task of neurons of output layer
        out_layer = tf.add(tf.matmul(layer_2, self.w3), self.b3)

        return out_layer


if __name__ == "__main__":
    try:
        TensorFlow()
    except Exception as e:
        raise e
