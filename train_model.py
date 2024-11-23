# Import necessary libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import optimizers

# Set up paths for training and testing datasets
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Data Preprocessing: Set up ImageDataGenerator for training and testing
# This includes data augmentation for training images and rescaling for both training and testing images

#data generator

train_datagen = ImageDataGenerator(

    rescale=1./255,               # Normalize the pixel values
    rotation_range=40,            # Random rotation of images
    width_shift_range=0.2,        # Random horizontal shifts
    height_shift_range=0.2,       # Random vertical shifts
    shear_range=0.2,              # Random shear transformation
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Random horizontal flips
    fill_mode='nearest'           # Fill pixels after transformations
     
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for testing

# Set up the training and testing data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,                   # Directory for training data
    target_size=(224, 224),      # Resize images to 224x224
    batch_size=32,               # Number of images per batch
    class_mode='binary'          # Binary classification (Waste vs. Non-Waste)
)

test_generator = test_datagen.flow_from_directory(
    test_dir,                    # Directory for testing data
    target_size=(224, 224),      # Resize images to 224x224
    batch_size=32,               # Number of images per batch
    class_mode='binary'          # Binary classification (Waste vs. Non-Waste)
)

# Build the CNN Model

model = Sequential([
    # First Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    # Second Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flatten the layers and feed them to a dense layer
    Flatten(),
    
    # Fully Connected Layer
    Dense(512, activation='relu'),
    
    # Output Layer with a single neuron for binary classification
    Dense(1, activation='sigmoid')
    
])

# Compile the model

model.compile(
    loss='binary_crossentropy',              # Use binary cross-entropy loss function
    optimizer=optimizers.Adam(),             # Adam optimizer
    metrics=['accuracy']                     # Evaluate the accuracy
)

# Train the model
history = model.fit(
    train_generator,                         # Training data
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of batches per epoch
    epochs=10,                               # Number of epochs to train
    validation_data=test_generator,          # Validation data
    validation_steps=test_generator.samples // test_generator.batch_size   # Number of batches per validation step
)

# Save the trained model
model.save('food_waste_identification_model.h5')

print("Model training complete and saved!")
