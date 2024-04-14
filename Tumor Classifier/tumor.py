import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def predict_image(file_path):
    # Load the pre-trained model
    new_model = load_model(os.path.join('models', 'imageclassifiertumor.h5'))

    # Read the selected image file
    image = cv2.imread(file_path)

    # Resize the image
    resized_image = tf.image.resize(image, (256, 256))

    # Perform prediction
    yyy = new_model.predict(np.expand_dims(resized_image / 255, 0))

    # Interpret predictions
    if 1 >= yyy > 0.5:
        prediction = 'Tumor'
    elif 0.000999 > yyy >= 0.9999:
        prediction = 'Invalid Image'
    else:
        prediction = 'Normal'

    return prediction

def browse_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])

    if file_path:
        # Predict the image class
        prediction = predict_image(file_path)

        # Show prediction result in a message box
        messagebox.showinfo("Prediction Result", f"Predicted class is {prediction}")

# Create a Tkinter window
root = tk.Tk()
root.title("Image Classifier")

# Create a button to browse for an image
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()