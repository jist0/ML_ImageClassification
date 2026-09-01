import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model("cifar10_cnn.keras")

# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️"
)

# Title
st.title("🖼️ CIFAR-10 Image Classification")

st.write(
    "Upload an image and the trained CNN model "
    "will predict its class."
)

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# Process image
if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize to CIFAR-10 input size
    image = image.resize((32, 32))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    prediction = model.predict(image_array)

    # Get predicted class
    predicted_class = np.argmax(prediction[0])

    # Get confidence
    confidence = np.max(prediction[0]) * 100

    # Display result
    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
