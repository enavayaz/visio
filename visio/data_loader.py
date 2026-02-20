import pandas as pd
from sklearn.model_selection import train_test_split
from utils import normalize, reshape
from tensorflow.keras.datasets import mnist

def load_csv(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    X = reshape(normalize(train.drop("label", axis=1).values))
    y = train["label"].values
    X_test = reshape(normalize(test.values))
    return X, y, X_test

def load_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = reshape(normalize(X_train))
    X_test  = reshape(normalize(X_test))
    return X_train, y_train, X_test, y_test

def split(X, y, test_size=0.1, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)