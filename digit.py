# Task: To identify handwritten numbers (0-9) with 70,000 samples.
# Task: To identify handwritten numbers.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Print dataset details
print("Length of the training dataset:", len(X_train))
print("Shape of the first training image:", X_train[0].shape)

# Display the first training image
plt.imshow(X_train[0])
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()

# Flatten the images
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

# Very simple neural network with no hidden layers.
# Sequential creates the neural network.
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=10)

