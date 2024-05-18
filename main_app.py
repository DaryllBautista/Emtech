import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import traceback

# Load the pre-trained model
try:
    model = load_model('best_modelnew.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app
st.title("Happy or Sad Face Predictor")

st.write("Upload an image to predict whether the face is happy or sad.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        result = 'happy' if prediction[0][0] > 0.5 else 'sad'

        st.write(f"The face in the image is **{result}**.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error(traceback.format_exc())
