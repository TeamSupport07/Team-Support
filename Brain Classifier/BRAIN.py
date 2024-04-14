import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
new_model = load_model(os.path.join('models', 'tanishk.h5'))
class_indices = {0: 'pituitary', 1: 'notumor', 2: 'meningioma', 3: 'glioma'}

def predict_image(image):
    # Resize the image
    resized_image = tf.image.resize(image, (150, 150))
    resized_input = resized_image / 255.0

    # Perform prediction
    predictions = new_model.predict(np.expand_dims(resized_input, 0))

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_indices[predicted_class_index]

    return predicted_class_label

def browse_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])

    if file_path:
        # Read the selected image file
        image = cv2.imread(file_path)

        # Convert image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Resize image
        resized_image = image_pil.resize((150, 150))

        # Convert image to NumPy array
        image_np = np.array(resized_image) / 255.0

        # Perform prediction
        prediction = predict_image(image_np)

        # Show prediction result in a message box
        messagebox.showinfo("Prediction Result", f"The predicted class label is: {prediction}")

# Create a Tkinter window
root = tk.Tk()
root.title("Brain Tumor Image Classifier")

# Create a button to browse for an image
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
