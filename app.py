from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
import io
from model import process_image
app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image"})
    
    image_data = image_file.read()
    
    try:
        result = process_image(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
