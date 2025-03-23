import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def fetch_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(f"Training images shape: {train_images.shape}")
    print(f"Testing images shape: {test_images.shape}")
    return (train_images, train_labels), (test_images, test_labels)
