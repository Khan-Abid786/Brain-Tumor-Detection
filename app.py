import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import os

# Function to load the model
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.write(f"Model file not found: {model_path}")
            return None
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.write(f"Error loading model: {e}")
        return None

# Load the trained model
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'brain_tumor_modelv1.h5')
model = load_model(model_path)

if model is not None:
    # Define a function to preprocess the uploaded image
    def preprocess_image(uploaded_file):
        # Load the image
        img = Image.open(uploaded_file).convert('RGB')  # Ensure the image is in RGB mode
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
        confidence = prediction[0][0]
        if confidence > 0.5:
            return f"Yes, Brain Tumor detected with {confidence * 100:.2f}% confidence"
        else:
            return f"No, Brain Tumor not detected with {(1 - confidence) * 100:.2f}% confidence"

    # Streamlit app
    #st.title("Welcome")
    #st.header("Brain Tumor Detection")
    #st.write("Upload an MRI image to detect brain tumor")
    st.markdown("""
    ---
    <body style = border:10px; color:White>
    <div >
    <h1 style='text-align: center; margin-top: 50px;color:Yellow'>Welcome</h1><br> <h2 style=' margin-top: 10px;color:Purple'>Brain Tumor Detection</h2>
    <br><h3 style = color:Maroon>Upload an MRI image to detect brain tumor</h3> </p>
    </body>
    </div>
    """, unsafe_allow_html=True)
    # File uploader
    uploaded_file = st.file_uploader("choose an image ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded MRI Image', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        # Make a prediction
        prediction = make_prediction(img_array)
        st.write(prediction)

else:
    st.write("Model could not be loaded. Please check the model file path and try again.")

# Footer
st.markdown("""
---
<div style='text-align: center; margin-top: 50px;color:Cyan'>
<p>Developed by<br>&nbsp <b><i>Khan Abid</i></b></p>
</div>
""", unsafe_allow_html=True)
