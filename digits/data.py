from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def preprocessing(train, split_train_size = 1/7):

    X_train = train.drop(["label"],
                         axis = 1)
    y_train = train["label"]

    # Reshape into right format vectors
    X_train = X_train.values.reshape(-1,28,28)

    # Apply ohe on labels
    y_train = to_categorical(y_train, num_classes = 10)

    # Split the train and the validation set for the fitting
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = split_train_size, random_state=42)

    return X_train, X_test, y_train, y_test


def load_data(data_file):

    """
    Return ready to use train and test with images and targets
    from_MNIST = True: load data from keras mnist dataset
    from_MNIST = False: load data from digit-recognizer dataset
    """

    # if from_MNIST:
    # # Load the data from mnist dataset (70k images)
    #     (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # else:
    #     # Load train from digit recognizer kaggle dataset (42k images)
    train = pd.read_csv(data_file)
    x_train, x_test, y_train, y_test = preprocessing(train)

    # Set pixel values between -1 and 1
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    nb_images_train = x_train.shape[0]
    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    x_train = x_train.reshape(nb_images_train, 784)
    return (x_train, y_train, x_test, y_test)
