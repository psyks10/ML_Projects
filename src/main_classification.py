#!/usr/bin/env

from logger import initialise_logger
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import math


class TensorFlow():

    def __init__(self):

        # Set seed
        np.random.seed(101)
        tf.compat.v1.set_random_seed(101)
        # Initialise logger
        self.logger = initialise_logger()
        self.logger.info("Hello")

        # Load data
        data = pd.read_csv('resources/ObesityDataSet_raw.csv', sep="\t",
                           names=["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"])
        print(data.head(10))


if __name__ == "__main__":
    try:
        TensorFlow()
    except Exception as e:
        raise e
