import os
import csv
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model from the h5 file
model = load_model('train_w_class_set')

# Path to the test directory
test_dir = 'compdata'

# List files in the test directory
test_files = os.listdir(test_dir)

# Initialize predictions list
predictions = []

# Iterate through test files
for file in test_files:
    # Skip markdown and HEIC files
    if file.endswith('.md') or file.endswith('.HEIC'):
        predictions.append((file, 0))  # Assign label 0
    else:
        # Load and preprocess the image
        img = Image.open(os.path.join(test_dir, file)).convert('RGB')
        img = img.resize((224, 224))  # Resize image to match model input shape
        img_array = np.array(img) / 255.0  # Normalize pixel values
        
        # Perform prediction
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        predicted_label = np.argmax(prediction)
        predictions.append((file, predicted_label))

# Save predictions to CSV file
with open('res.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for prediction in predictions:
        writer.writerow(prediction)
