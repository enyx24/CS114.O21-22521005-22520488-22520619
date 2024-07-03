import cv2
import numpy as np
import os
import csv
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

def preprocess_image_with_edges(image):
    median_filtered = cv2.medianBlur(image, 5)
    edges = cv2.Canny(median_filtered, 50, 150)
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    binary = cv2.adaptiveThreshold(median_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_or(binary, dilated_edges)
    return combined

def extract_hog_features_opencv(images):
    hog = cv2.HOGDescriptor(_winSize=(28,28),
                            _blockSize=(14,14),
                            _blockStride=(7,7),
                            _cellSize=(14,14),
                            _nbins=9)
    hog_features = []
    for image in images:
        preprocessed_image = preprocess_image_with_edges(image)
        hog_descriptor = hog.compute(preprocessed_image)
        hog_features.append(hog_descriptor.flatten())
    return hog_features

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (28, 28))  # Resize to MNIST size
            images.append(img_resized)
            filenames.append(filename)
    return images, filenames

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the MNIST images for CNN
x_train_cnn = np.stack([cv2.resize(img, (32, 32)) for img in x_train], axis=0)
x_test_cnn = np.stack([cv2.resize(img, (32, 32)) for img in x_test], axis=0)
x_train_cnn = np.expand_dims(x_train_cnn, axis=-1)
x_test_cnn = np.expand_dims(x_test_cnn, axis=-1)
x_train_cnn = np.repeat(x_train_cnn, 3, axis=-1)  # Convert grayscale to RGB
x_test_cnn = np.repeat(x_test_cnn, 3, axis=-1)

# Normalize the data
x_train_cnn = x_train_cnn / 255.0
x_test_cnn = x_test_cnn / 255.0

# Load pretrained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Extract features from an intermediate layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Extract features
x_train_features = model.predict(x_train_cnn)
x_test_features = model.predict(x_test_cnn)

# Flatten the features
x_train_features = x_train_features.reshape((x_train_features.shape[0], -1))
x_test_features = x_test_features.reshape((x_test_features.shape[0], -1))

# Preprocess the MNIST images
x_train_preprocessed = [preprocess_image_with_edges(img) for img in x_train]
x_test_preprocessed = [preprocess_image_with_edges(img) for img in x_test]

# Extract HOG features using OpenCV
x_train_hog = extract_hog_features_opencv(x_train_preprocessed)
x_test_hog = extract_hog_features_opencv(x_test_preprocessed)

# Combine CNN features with HOG features
x_train_combined = np.hstack((x_train_features, x_train_hog))
x_test_combined = np.hstack((x_test_features, x_test_hog))

# Train SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(x_train_combined, y_train)

# Evaluate on the test set
y_pred = clf.predict(x_test_combined)
print("Accuracy on MNIST test set:", accuracy_score(y_test, y_pred))

# Load and preprocess images from the folder "compdata"
compdata_folder = 'compdata'
compdata_images, compdata_filenames = load_images_from_folder(compdata_folder)

# Preprocess the compdata images
compdata_preprocessed = [preprocess_image_with_edges(img) for img in compdata_images]

# Extract HOG features for compdata images
compdata_hog = extract_hog_features_opencv(compdata_preprocessed)

# Preprocess for CNN
compdata_cnn = np.stack([cv2.resize(img, (32, 32)) for img in compdata_images], axis=0)
compdata_cnn = np.expand_dims(compdata_cnn, axis=-1)
compdata_cnn = np.repeat(compdata_cnn, 3, axis=-1)
compdata_cnn = compdata_cnn / 255.0

# Extract CNN features
compdata_features = model.predict(compdata_cnn)
compdata_features = compdata_features.reshape((compdata_features.shape[0], -1))

# Combine CNN and HOG features
compdata_combined = np.hstack((compdata_features, compdata_hog))

# Predict using the trained SVM model
compdata_predictions = clf.predict(compdata_combined)

# Save the results to res.csv
with open('res.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label"])
    for filename, prediction in zip(compdata_filenames, compdata_predictions):
        writer.writerow([filename, prediction])

print("Predictions saved to res.csv")
