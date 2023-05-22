import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Parse command-line arguments
parser = argparse.ArgumentParser(description='MNIST model training and prediction')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--image_dir', type=str, default='images', help='directory containing input images')
args = parser.parse_args()

# Create the directory if it doesn't exist
os.makedirs(args.image_dir, exist_ok=True)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_test, y_test))

# Save the model
model.save('/cnvrg/mnist_model.h5')

# Load the model
model = tf.keras.models.load_model('/cnvrg/mnist_model.h5')

# Function to predict the digit in an image
def predict_digit(image):
    # Preprocess the image
    image = np.array(image).reshape(-1, 28, 28, 1).astype('float32') / 255.0
