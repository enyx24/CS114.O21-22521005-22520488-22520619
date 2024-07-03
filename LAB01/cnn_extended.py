import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2

# Define a new model architecture
def create_model(input_shape=(100, 100, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images_resized = np.array([cv2.resize(img, (100, 100)) for img in train_images])
test_images_resized = np.array([cv2.resize(img, (100, 100)) for img in test_images])
train_images_resized = train_images_resized / 255.0
test_images_resized = test_images_resized / 255.0
train_images_resized = train_images_resized.reshape((-1, 100, 100, 1))
test_images_resized = test_images_resized.reshape((-1, 100, 100, 1))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the new model
model = create_model()

# Train the model
model.fit(train_images_resized, train_labels, epochs=5, batch_size=32, validation_data=(test_images_resized, test_labels))

# Save the new model
model.save('model_100x100.h5')
