import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models, optimizers

# Define the batch size
batch_size = 8  # You can adjust this to your desired batch size

# Define the directory containing the data
data_dir = 'unioned_data'

# Load InceptionV3 model without the top classification layer
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data generator for training without shear and horizontal flip
train_datagen = ImageDataGenerator(
    rescale=1./255,   # Rescale pixel values to [0, 1]
    zoom_range=0.2,   # Zoom range for random transformations
    validation_split=0.2)   # Split 20% of the data for validation


# Create the data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,  # Use the reduced batch size
    class_mode='sparse',
    subset='training')  # Use the training subset

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,  # Use the reduced batch size
    class_mode='sparse',
    subset='validation')  # Use the validation subset

# Function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Get class labels from the generator
class_labels = list(train_generator.class_indices.keys())

# Plot 10 random images
random.seed(42)
plt.figure(figsize=(10, 10))
for i in range(10):
    random_class_index = i
    filepaths_for_random_class = [train_generator.filepaths[i] for i, c in enumerate(train_generator.classes) if c == random_class_index]
    # Lấy ngẫu nhiên một đường dẫn từ danh sách filepaths
    random_img_path = random.choice(filepaths_for_random_class)
    img_array = preprocess_image(random_img_path)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img_array)
    plt.title(random_class_index)
    plt.axis('off')
plt.show()

# Count number of images per class in training set
train_counter = Counter(train_generator.classes)
print("Number of images per class in training set:")
for class_idx, count in train_counter.items():
    print(f"Class {class_idx}: {count} images")

# Count number of images per class in validation set
val_counter = Counter(validation_generator.classes)
print("\nNumber of images per class in validation set:")
for class_idx, count in val_counter.items():
    print(f"Class {class_idx}: {count} images")

def create_inception_model(input_shape, num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the InceptionV3 model
model = create_inception_model(input_shape=(224, 224, 3), num_classes=train_generator.num_classes)
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=1)

# Save the trained model
model.save('train_w_class_set')
