#!/usr/bin/env
import io
import shutil
import zipfile

import requests

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score


class TensorFlow():

    def __init__(self):

        # Set seed
        self.seed = 101
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Network parameters
        n_hidden1 = 10
        n_hidden2 = 10
        n_hidden3 = 10
        n_input = 16
        n_output = 1

        # Learning parameters
        learning_constant = 0.02
        number_epochs = 1000
        batch_size = 50

        # Defining the input and the output
        X = tf.placeholder("float", [None, n_input], name="X")
        Y = tf.placeholder("float", [None, n_output], name="Y")

        # DEFINING WEIGHTS AND BIASES

        # Biases first hidden layer
        self.b1 = tf.Variable(tf.random_normal([n_hidden1]))
        # Biases second hidden layer
        self.b2 = tf.Variable(tf.random_normal([n_hidden2]))
        # Biases third layer
        self.b3 = tf.Variable(tf.random_normal([n_hidden3]))
        # Biases output layer
        self.b4 = tf.Variable(tf.random_normal([n_output]))

        # Weights connecting input layer with first hidden layer
        self.w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
        # Weights connecting first hidden layer with second hidden layer
        self.w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
        # Weights connecting second hidden layer with third layer
        self.w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))
        # Weights connecting third hidden layer with output layer
        self.w4 = tf.Variable(tf.random_normal([n_hidden3, n_output]))

        # Create model
        neural_network = self.multilayer_perceptron(X)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
        # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Load the data
        X_data, y_data, X_train, X_test, y_train, y_test = self.load_obesity_data()

        with tf.Session() as sess:

            sess.run(init)

            validation_data_accuracies = []
            validation_data_costs = []

            # K-Fold
            k_fold = KFold(n_splits=10, random_state=self.seed)
            i = 0
            for train_indices, validation_indices in k_fold.split(X_train):
                i += 1
                print(f"\nFold no. {i}")

                X_train_fold, y_train_fold = X_train[train_indices], y_train[train_indices]
                X_validation_fold, y_validation_fold = X_train[validation_indices], y_train[validation_indices]

                for epoch in range(number_epochs):
                    _, c = sess.run([optimizer, loss_op], feed_dict={X: X_train_fold, Y: y_train_fold})

                # Evaluation
                pred = neural_network

                accuracy = self.root_mean_squared_error(pred, Y)

                print(f'RMSE: {accuracy.eval({X: X_validation_fold, Y: y_validation_fold})}')
                print(f"Cost: {c}")
                validation_data_accuracies.append(np.mean(accuracy.eval({X: X_validation_fold, Y: y_validation_fold})))
                validation_data_costs.append(c)


            # Evaluation - training data
            print(f'\nK-Fold RMSE: {validation_data_accuracies}')
            print(f'Average K-Fold RMSE: {np.mean(validation_data_accuracies)}')


            # Evaluation - test data
            print('\nEvaluation of test data')
            accuracy = self.root_mean_squared_error(pred, Y)
            print(f'RMSE: {accuracy.eval({X: X_test, Y: y_test})}')

            # plt.plot(y_test[0:10], 'ro', pred.eval({X:X_test})[0:10], 'bo')
            # plt.ylabel('some numbers')
            # plt.show()


    def root_mean_squared_error(self, y_true, y_pred):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

    def generate_summary(self, X_placeholder, Y_placeholder, X_values, y_values, accuracy, neural_network, session):
        result = pd.DataFrame()
        result['label'] = pd.DataFrame(y_values).idxmax(axis=1).values
        result['label_encoded'] = list(y_values)
        result['output'] = list(neural_network.eval({X_placeholder: X_values}))
        result['prediction'] = list(session.run(tf.argmax(neural_network, axis=1), feed_dict={X_placeholder: X_values}))
        result['accuracy'] = list(accuracy.eval({X_placeholder: X_values, Y_placeholder: y_values}))
        return result

    def load_obesity_data(self):

        file = 'ObesityDataSet_raw_and_data_sinthetic%20%282%29.zip'
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00544/'

        # Download zip file into folder 'data' and extract files
        zipfile.ZipFile(io.BytesIO(requests.get(path + file).content)).extractall("data")
        self.data = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv', skiprows=1, sep=',',
                                names=["Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC",
                                       "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS",
                                       "NObeyesdad"])

        # After extracting data from csv delete folder 'data'
        dir_path = 'data'
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))

        # Separate features and label
        y_data = pd.DataFrame(self.data['NObeyesdad'])
        X_data = self.data.drop('NObeyesdad', axis=1)

        # Encode the label
        d = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6}
        y_data['NObeyesdad'] = y_data['NObeyesdad'].map(d)

        # Encoding categorical features
        categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC",
                               "MTRANS"]
        for column in categorical_columns:
            print(f"Encoding of {column}: \n{dict(enumerate(X_data[column].astype('category').cat.categories))}\n")
            X_data[column] = X_data[column].astype('category').cat.codes

        # Split data in Training 60%, Test 20% and Validation 20%
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.40, random_state=self.seed)

        # Convert to arrays
        X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values,

        return X_data, y_data, X_train, X_test, y_train, y_test

    def multilayer_perceptron(self, input_d):
        # Task of neurons of first hidden layer
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, self.w1), self.b1))
        # Task of neurons of second hidden layer
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        # Task of neurons of third hidden layer
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.w3), self.b3))
        # Task of neurons of output layer
        out_layer = tf.add(tf.matmul(layer_3, self.w4), self.b4)

        return out_layer


if __name__ == "__main__":
    try:
        TensorFlow()
    except Exception as e:
        raise e
