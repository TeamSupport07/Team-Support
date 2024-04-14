import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained image classification model
def load_image_classifier(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Load the model
model_path = 'models/tanishk.h5'  # Update with the correct path to your model
model = load_image_classifier(model_path)

# Define class labels
class_indices = {0: 'pituitary', 1: 'notumor', 2: 'meningioma', 3: 'glioma'}

# Route handler for the root URL
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file from the request
        file = request.files['file']

        # Read the image file
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize the image
        resized_image = tf.image.resize(img, (150, 150))

        # Preprocess the image
        preprocessed_image = resized_image / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        # Make prediction using the pre-trained model
        predictions = model.predict(preprocessed_image)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class_label = class_indices[predicted_class_index]

        # Render the prediction result template with the predicted class
        return render_template('prediction_result.html', prediction=predicted_class_label)

    except Exception as e:
        # Handle errors gracefully and return an error response
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the image. Please try again.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000,debug=True)
