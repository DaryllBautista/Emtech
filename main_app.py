import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import subprocess

# Set page layout
st.set_page_config(
    page_title="Happy or Sad Classification",
    page_icon=":smiley:",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #000000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a title with a custom font and color
st.title("Happy or Sad Classification")
st.markdown(
    """
    <style>
        .big-font {
            font-size: 24px !important;
            color: #3498db !important;
        }
        .highlight {
            background-color: #3498db;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .result {
            background-color: #3498db;
            padding: 10px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .result-text {
            color: #ffffff !important;
        }
        .white-text {
            color: #ffffff !important;
        }
        .footer-text {
            color: #ffffff !important;
            font-style: italic;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Text to display
made_by_text = "This was made by:"
submitted_to_text = "Submitted to Engr Roman Richard."

# Display "This was made by:" with a blue background and white text
st.markdown(f'<div class="highlight"><p class="big-font white-text">{made_by_text}</p></div>', unsafe_allow_html=True)

# Names to display
names = ["Bautista, Daryll Milton Victor E.", "Tavares, Nicole Ann"]

# Display names with a blue background and white text
for name in names:
    st.markdown(f'<p class="big-font white-text">{name}</p>', unsafe_allow_html=True)

st.markdown(f'<p class="big-font white-text">{submitted_to_text}</p>', unsafe_allow_html=True)

# File uploader widget to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Function to perform image classification using TensorFlow
def classify_image(image):
    try:
        # Load the trained model (replace with your own model)
        model = tf.keras.models.load_model("best_modelnew.h5")

        # Preprocess the image
        img_array = np.array(image)
        img_array = tf.image.resize(img_array, (64, 64))  # Resize the image to match model's expected input size
        img_array = tf.expand_dims(img_array, 0)  # Add a batch dimension
        img_array = img_array / 255.0  # Normalize the input image

        # Debug: Output preprocessed image shape
        st.write("Preprocessed image shape:", img_array.shape)

        # Make predictions
        predictions = model.predict(img_array)

        # Debug: Output raw predictions
        st.write("Raw Predictions:", predictions)

        return predictions

    except Exception as e:
        st.error(f"Error during image classification: {e}")
        return None

# Display the uploaded image and perform classification
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Add a header with a blue background and white text
    st.markdown('<div class="highlight"><p class="big-font white-text">Uploaded Image</p></div>', unsafe_allow_html=True)

    # Display the image
    st.image(image, use_column_width=True)

    # Call the classification function
    predictions = classify_image(image)

    if predictions is not None:
        # Display the classification results with a blue background
        st.markdown('<div class="result"><p class="big-font result-text">Prediction Results</p></div>', unsafe_allow_html=True)

        # Extracting class labels
        class_labels = ["Sad", "Happy"]

        # Finding the predicted class
        predicted_class = "Happy" if predictions[0][0] >= 0.5 else "Sad"

        # Display the result with white text on a blue background
        st.markdown(f'The model predicts: <span class="big-font result-text">{predicted_class}</span>', unsafe_allow_html=True)

# Link to open the app in Colab
colab_link = "<a href=\"https://colab.research.google.com/github/DaryllBautista/Emtech/blob/main/final_requirement_streamlit.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
st.markdown(colab_link, unsafe_allow_html=True)

# Run Streamlit app in the background
streamlit_command = "streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py"
subprocess.Popen(streamlit_command, shell=True)
