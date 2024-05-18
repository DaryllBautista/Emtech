# Final-require-for-emtech
**1.Explanation of your project.**
Our project consists of two separate notebooks. The first notebook, "final_requirement.ipynb," trains a convolutional neural network (CNN) model using Keras to classify images of happy and sad faces. It evaluates the model's performance on a test set and makes predictions on individual images. After training the model, the second notebook, "main_app.py," creates a Streamlit application for image classification. Users can upload an image, and the trained model predicts whether the uploaded image depicts a happy or sad face. Additionally, the app displays information about the creators and provides a link to run the app in Google Colab.


**2. What problem is it solving.**
Our code aims to address a happy or sad image classification task using convolutional neural networks (CNNs). It starts by preparing and preprocessing the image data, then constructs and trains a CNN model using Keras. During training, the model's best version is saved based on validation accuracy. After training, the model's performance is evaluated on a separate test set. Finally, a Streamlit application is created to allow users to upload images for real-time classification using the trained model, providing a simple interface for users to interact with the classification system.
