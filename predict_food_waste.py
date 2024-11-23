import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('D:/food-waste-identification/food_waste_identification_model.h5')


# Function to predict whether the food is waste or non-waste
def predict_food(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to match the input size of the model
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
   
    # Normalize the image array (important if your model was trained with normalized images)
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]

    # Make a prediction
    prediction = model.predict(img_array)

    # Interpret the result
    if prediction[0] > 0.5:
        print("This image is classified as Waste.")
    else:
        print("This image is classified as Non-Waste.")


# predict_food('rotten_ban.jpg') 
# predict_food('fresh_ban.jpg')  

# predict_food('fresh_mango.jpg')  
# predict_food('rotten_mango.jpg') 

# predict_food('fresh_grapes.jpg') 
predict_food('rot_grape.jpg') 
