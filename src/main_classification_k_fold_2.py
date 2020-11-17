#!/usr/bin/env

import functions.logger as logger
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, fbeta_score


class TensorFlow():

    def __init__(self):

        # Set seed
        self.seed = 101
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Network parameters
        n_hidden1 = 15
        n_hidden2 = 15
        n_input = 9
        n_output = 2

        # Learning parameters
        learning_constant = 0.05
        number_epochs = 1000

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
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Load the data
        X_data, y_data, X_train, X_test, y_train, y_test, label = self.load_breast_cancer_data()

        with tf.Session() as sess:

            sess.run(init)

            training_data_costs = []

            validation_data_accuracies = []
            validation_data_costs = []

            ##################################
            # Normal split- Used for experimenting with hyperparameters
            ##################################
            # method = 'Normal split'
            #
            # # Split data in Training 70%, Validation 30%
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
            #
            # for epoch in range(number_epochs):
            #
            #     if epoch % 100 == 0:
            #         print(f"\nEpoch no. {epoch}")
            #
            #     _, c = sess.run([optimizer, loss_op], feed_dict={X: X_train, Y: y_train})
            #     training_data_costs.append(c)
            #
            #     # Evaluation
            #     pred = tf.nn.softmax(neural_network)  # Apply softmax to logits
            #     estimated_class = tf.argmax(pred, axis=1)
            #     correct_prediction = tf.equal(tf.argmax(pred, axis=1), pd.DataFrame(y_val).idxmax(axis=1).values)
            #
            #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #     if epoch % 100 == 0:
            #         print(f'Class estimation: {sess.run(estimated_class, feed_dict={X: X_val})}')
            #         print(f'Correct predictions: {sess.run(correct_prediction, feed_dict={X: X_val}).sum()} '
            #               f'out of {len(y_val)}')
            #         print(f'Accuracy: {accuracy.eval({X: X_val})}')
            #         print(f"Cost: {c}")
            #     validation_data_accuracies.append(accuracy.eval({X: X_val}))
            #     validation_data_costs.append(loss_op.eval({X: X_val, Y: y_val}))

            ##################################
            # K-Fold
            ##################################
            method = 'K-Fold'

            k_fold = KFold(n_splits=10, random_state=self.seed)
            i = 0
            for train_indices, validation_indices in k_fold.split(X_train):
                i += 1
                print(f"\nFold no. {i}")

                X_train_fold, y_train_fold = X_train[train_indices], y_train[train_indices]
                X_validation_fold, y_validation_fold = X_train[validation_indices], y_train[validation_indices]

                for epoch in range(number_epochs):
                    _, c = sess.run([optimizer, loss_op], feed_dict={X: X_train_fold, Y: y_train_fold})
                training_data_costs.append(c)

                # Evaluation - Validation data
                pred = tf.nn.softmax(neural_network)  # Apply softmax to logits
                estimated_class = tf.argmax(pred, axis=1)
                correct_prediction = tf.equal(tf.argmax(pred, axis=1), pd.DataFrame(y_validation_fold).idxmax(axis=1).values)

                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                y_true = pd.DataFrame(y_validation_fold).idxmax(axis=1).values
                y_pred = sess.run(estimated_class, feed_dict={X: X_validation_fold})
                print(f'Correct Class: {y_true}')
                print(f'Class prediction: {y_pred}')
                print(f'Correct predictions: {sess.run(correct_prediction, feed_dict={X: X_validation_fold}).sum()} '
                      f'out of {len(y_validation_fold)}')
                print(f'Accuracy: {accuracy.eval({X: X_validation_fold})}')
                print(f"Cost: {c}")
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                print(f'True negatives: {tn}\nFalse positives: {fp}\nFalse negatives: {fn}\nTrue positives: {tp}')
                print(f'F1 Score (beta = 1): {fbeta_score(y_true, y_pred, 1)}')
                print(f'F1 Score (beta = 2): {fbeta_score(y_true, y_pred, 2)}')
                validation_data_accuracies.append(accuracy.eval({X: X_validation_fold}))
                validation_data_costs.append(loss_op.eval({X: X_validation_fold, Y: y_validation_fold}))

            # Evaluation - Training data
            print(f'\n{method} accuracy: {validation_data_accuracies}')
            print(f'Average {method} accuracy: {np.mean(validation_data_accuracies)}')
            print(f'Average {method} cost: {np.mean(validation_data_costs)}')
            print(f'Last {method} accuracy: {validation_data_accuracies[-1]}')
            print(f'Last {method} cost: {validation_data_costs[-1]}')

            plt.plot(validation_data_costs, 'b.', label="Validation")
            plt.plot(training_data_costs, 'r.', label="Training")
            plt.title(f'Cost per Epoch - Learning Rate: {learning_constant}')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend(loc="upper right")
            plt.show()

            # Evaluation - test data
            estimated_class = tf.argmax(pred, axis=1)
            correct_prediction = tf.equal(tf.argmax(pred, axis=1), pd.DataFrame(y_test).idxmax(axis=1).values)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            print('\nEvaluation of test data')
            y_true = pd.DataFrame(y_test).idxmax(axis=1).values
            y_pred = sess.run(estimated_class, feed_dict={X: X_test})
            print(f'Correct Class: {y_true}')
            print(f'Class prediction: {y_pred}')
            print(f'Correct predictions: {sess.run(correct_prediction, feed_dict={X: X_test}).sum()} '
                  f'out of {len(X_test)}')
            print(f'Accuracy: {accuracy.eval({X: X_test})}')
            print(f'Cost: {loss_op.eval({X: X_test, Y: y_test})}')
            print(f'Confusion matrix:\n {confusion_matrix(y_true, y_pred, labels=[0,1])}')
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print(f'True negatives: {tn}\nFalse positives: {fp}\nFalse negatives: {fn}\nTrue positives: {tp}')
            print(f'F1 Score (beta = 1): {fbeta_score(y_true, y_pred, 1)}')
            print(f'F1 Score (beta = 2): {fbeta_score(y_true, y_pred, 2)}')

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

        # Split data in Training 80%, Test 20%
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=self.seed)

        # Convert to arrays
        X_train, X_test, y_train, y_test, = X_train.values, X_test.values, y_train.values, y_test.values,

        return X_data, y_data, X_train, X_test, y_train, y_test, label

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
