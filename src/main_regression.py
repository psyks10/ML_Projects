import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from src.logger import initialise_logger


class TensorFlow:

    def __init__(self):
        import requests
        import zipfile
        import io
        import shutil

        # Set seed
        self.seed = 101
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Initialise logger
        self.logger = initialise_logger()

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

        # Network parameters
        n_hidden1 = 10
        n_hidden2 = 10
        n_hidden3 = 10
        n_input = 16
        n_output = 7

        # Learning parameters
        learning_constant = 0.3
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
        # Biases second hidden layer
        self.b3 = tf.Variable(tf.random_normal([n_hidden3]))
        # Biases output layer
        self.b4 = tf.Variable(tf.random_normal([n_output]))

        # Weights connecting input layer with first hidden layer
        self.w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
        # Weights connecting first hidden layer with second hidden layer
        self.w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
        # Weights connecting first hidden layer with second hidden layer
        self.w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))
        # Weights connecting second hidden layer with output layer
        self.w4 = tf.Variable(tf.random_normal([n_hidden3, n_output]))

        # Create model
        neural_network = self.multilayer_perceptron(X)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Load the data
        X_data, y_data, X_train, X_test, X_val, y_train, y_test, y_val, label = self.load_obesity_data()

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(number_epochs):

                avg_cost = 0.0
                total_batch = int(len(X_train) / batch_size)
                X_train_batches = np.array_split(X_train, total_batch)
                y_train_batches = np.array_split(y_train, total_batch)

                # Mini-batch training
                for X_batch, y_batch in zip(X_train_batches, y_train_batches):
                     sess.run(optimizer, feed_dict={X: X_batch, Y: y_batch})

                # Display the epoch
                if epoch % (number_epochs/10) == 0:
                    print("Epoch:", '%d' % epoch)
                    pred = sess.run(neural_network, feed_dict={X: X_test})
                    #print("prediction:", np.argmax(pred, axis=1))
                    #print("y predicts:", np.argmax(y_test, axis=1))
                    print("Cost:", sess.run(loss_op, feed_dict={X: X_test, Y: y_test}))
                    accuracy = tf.keras.losses.MSE(pred, Y)
                    print("Accuracy:", sess.run(accuracy, feed_dict={X: X_test, Y: y_test}))
                    print("______________________________________")


            # Test model
            pred = (neural_network)  # Apply softmax to logits
            accuracy = tf.keras.losses.MSE(pred, Y)
            print("Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
            print("Prediction:", pred.eval({X: X_train}))
            output = neural_network.eval({X: X_train})

            plt.plot(y_train[0:10], 'ro', output[0:10], 'bo')
            plt.ylabel('some numbers')
            plt.show()

            # Training accuracy
            estimated_class = tf.argmax(pred, 1)  # +1e-50-1e-50
            correct_prediction1 = tf.equal(estimated_class, label)
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
            #print("correct_prediction1: ", correct_prediction1)
            #print("estimated_class: ", estimated_class)
            print(accuracy1.eval({X: X_data}))

            # Validation accuracy
            estimated_class = tf.argmax(pred, axis=1)  # +1e-50-1e-50
            correct_prediction1 = tf.equal(estimated_class, pd.DataFrame(y_val).idxmax(axis=1).values)
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
            #print("correct_prediction1: ", correct_prediction1)
            #print("estimated_class: ", estimated_class)
            print("accuracy1", accuracy1.eval({X: X_val}))






    def load_obesity_data(self):
        # Separate features and label
        y_data = pd.DataFrame(self.data['NObeyesdad'])
        X_data = self.data.drop('NObeyesdad', axis=1)
        # Set label
        label = self.data['NObeyesdad'].astype('category').cat.codes
        label = label.values
        # Encode the label using dummies
        y_data = pd.get_dummies(y_data)

        # Encoding categorical features
        categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC",
                               "MTRANS"]
        for column in categorical_columns:
            print(f"Encoding of {column}: \n{dict(enumerate(X_data[column].astype('category').cat.categories))}\n")
            X_data[column] = X_data[column].astype('category').cat.codes

        X_data = ((X_data - X_data.min()) / (X_data.max() - X_data.min()))
        y_data = ((y_data - y_data.min()) / (y_data.max() - y_data.min()))

        # Split data in Training 60%, Test 20% and Validation 20%
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.40, random_state=self.seed)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=self.seed)

        # Convert to arrays
        X_train, X_test, X_val, y_train, y_test, y_val = X_train.values, X_test.values, X_val.values, y_train.values, \
                y_test.values, y_val.values
        return X_data, y_data, X_train, X_test, X_val, y_train, y_test, y_val, label

    def multilayer_perceptron(self, input_d):
        # Task of neurons of first hidden layer
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(input_d, self.w1), self.b1))
        # Task of neurons of second hidden layer
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        # Task of neurons of third hidden layer
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, self.w3), self.b3))
        # Task of neurons of output layer
        out_layer = tf.add(tf.matmul(layer_3, self.w4), self.b4)
        return out_layer


if __name__ == "__main__":
    try:
        TensorFlow()
    except Exception as e:
        raise e
