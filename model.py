from keras.models import model_from_json

# Load the JSON file containing the model architecture
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Reconstruct the model from the JSON
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights("model.weights.h5")

import numpy as np
from PIL import Image
import io



def process_image(image_data):
    classes = ['algal leaf in tea',
 'bird eye spot in tea',
 'brown blight in tea',
 'healthy tea leaf',
 'red leaf spot in tea']
    # Convert image data to numpy array
    image = Image.open(image_data)
    image = image.resize((224, 224))
    image_array = np.array(image)
    # Divide by 255 to normalize pixel values
    processed_image = image_array
    pred = loaded_model.predict(np.expand_dims(processed_image, axis=0))
    # print(pred)
    return classes[np.argmax(pred)]