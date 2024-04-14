import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template  # Import render_template

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

# Define class labels (replace with your actual class labels)
class_labels = {
    0: 'Normal',
    1: 'Pneumonia'
}

# Load the model
model_path = 'models/imageclassifier (4).h5'  # Assuming the model is in 'models' directory
model = load_image_classifier(model_path)

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

        # Preprocess the image (resize, normalize)
        preprocessed_image = cv2.resize(img, (256, 256))
        preprocessed_image = preprocessed_image.astype('float32') / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        # Make prediction using the pre-trained model
        prediction = model.predict(preprocessed_image)[0]
        yhat = prediction[np.argmax(prediction)]
        
        # Check the prediction probability and print the predicted class accordingly
        if 1 >= yhat > 0.5:
            predicted_class = 'Pneumonia'
        elif 0.000999 > yhat >= 0.9999:
            return jsonify({'error': 'Please insert a valid image'}), 400
        else:
            predicted_class = 'Normal'

        # Render the prediction result template with the predicted class
        return render_template('prediction_result.html', prediction=predicted_class)

    except Exception as e:
        # Handle errors gracefully and return an error response
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the image. Please try again.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
