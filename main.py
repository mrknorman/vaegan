import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import fashion_mnist

#Load Dataset:

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.



