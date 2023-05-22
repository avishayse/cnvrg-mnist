import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def load_model():
    # Load the model
    model = tf.keras.models.load_model('/cnvrg/mnist_model.h5')
    return model

def preprocess_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_digit(model, img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    # Predict the digit
    prediction = model.predict(img_array)
    # Get the predicted digit
    digit = np.argmax(prediction)
    return digit.item()  # Convert int64 to standard Python integer

def predict(img_path):
    # Load the model
    model = load_model()
    # Predict the digit in the image
    predicted_digit = predict_digit(model, img_path)
    return {'predicted_digit': predicted_digit}
