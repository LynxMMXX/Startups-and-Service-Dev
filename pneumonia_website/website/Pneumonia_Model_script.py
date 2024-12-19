# %%
import os
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend like Agg
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.set_visible_devices([], 'GPU') 
from tensorflow import keras

os.listdir("./pneumonia_website/website/chest_xray")

#imports necessary for the Ai to work

# %%
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#importing the tools required for python to recognize the model

# %%
new_model = tf.keras.models.load_model('./pneumonia_website/website/my_checkpoint_Pnuemonia.keras')

# Show the model architecture
new_model.summary()

#loading the weights and biases of the pre-trained model and providing a summary

# %%

image_generator = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    shear_range=0,
    zoom_range=0,
    samplewise_center=True,
    samplewise_std_normalization=True
)

#doing a simple normalization and centering of the image to make it easier for the model to read

# %%
pnmoniatestdir = "./pneumonia_website/website/test_images_Pnmonia"
PnmoniaTest = image_generator.flow_from_directory(pnmoniatestdir, 
                                            batch_size=1, 
                                            shuffle=False, 
                                            class_mode='binary',
                                            target_size=(180, 180))

#finding the images directory
'''
def runImageTesting():
    model_prediction = new_model.predict(PnmoniaTest)

    plt.subplot(3, 3, 1)
    img = plt.imread(os.path.join("./pneumonia_website/website/test_images_Pnmonia/uploaded_images", os.listdir("./pneumonia_website/website/test_images_Pnmonia/uploaded_images")[0]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    return str(model_prediction[0]*100)

image_generator = ImageDataGenerator(
    samplewise_center=True,  # Center each image by subtracting its mean
    samplewise_std_normalization=True  # Normalize each image by dividing by its standard deviation
)'''

def runImageTesting(input_image):
    # Step 1: Open the image using PIL and convert to numpy array
    image_array = np.array(input_image, dtype=np.float32)  # Ensure the array is float32
    
    # Step 2: Convert grayscale to RGB (if necessary)
    if len(image_array.shape) == 2:  # Image is grayscale (H, W)
        image_array = np.expand_dims(image_array, axis=-1)  # Shape becomes (H, W, 1)
        image_array = np.repeat(image_array, 3, axis=-1)  # Convert to (H, W, 3)
    
    # Step 3: Resize the image to (180, 180)
    image_array = tf.image.resize(image_array, (180, 180))  # Resize to (180, 180, 3)

    # Step 4: Ensure the image array is writable (make a copy if necessary)
    image_array = np.copy(image_array)  # Make a writable copy

    # Step 5: Add batch dimension (model expects a batch, even for a single image)
    image_array = np.expand_dims(image_array, axis=0)  # Shape becomes (1, 180, 180, 3)

    # Step 6: Standardize the image using the ImageDataGenerator
    image_array = image_generator.standardize(image_array)  # Standardize the image

    # Step 7: Predict the class using the pre-trained model
    model_prediction = new_model.predict(image_array)

    # Step 8: Return the prediction result as a percentage
    return str(model_prediction[0][0] * 100)  # Assuming output is a probability (0-1)
