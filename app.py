import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_modelv2.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    # Load the image
    img = Image.open(uploaded_file)
    # Resize the image to the target size (224, 224)
    img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale the image
    img_array = img_array / 255.0
    return img_array

# Define a function to make predictions
def make_prediction(img_array):
    # Predict the class of the image
    prediction = model.predict(img_array)
    # Map the prediction to the class label
    if prediction > 0.5:
        return "Yes, Brain Tumor detected"
    else:
        return "No, Brain Tumor not detected"

# Streamlit app
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to detect brain tumor")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make a prediction
    prediction = make_prediction(img_array)
    st.write(prediction)
