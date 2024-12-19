# %%
import os
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend like Agg
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
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

def runImageTesting():
    model_prediction = new_model.predict(PnmoniaTest)

    plt.subplot(3, 3, 1)
    img = plt.imread(os.path.join("./pneumonia_website/website/test_images_Pnmonia/uploaded_images", os.listdir("./pneumonia_website/website/test_images_Pnmonia/uploaded_images")[0]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    return str(model_prediction[0]*100)

#using a for loop to check the number of images and giving a prediction for each image

# %%
#add new images to the "test_images_Pnmonia" directory to do more predictions

