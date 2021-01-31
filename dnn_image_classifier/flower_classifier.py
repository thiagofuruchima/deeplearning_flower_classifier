import os
# prevent the TensorFlow warnings from showing in the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
from PIL import Image

def process_image(image):
    """ Process the image file for use in TensorFlow """
    
    image_size = 224
    tf_image = tf.cast(image, tf.float32)
    tf_image = tf.image.resize(tf_image, (image_size, image_size))
    tf_image /= 255
    
    return tf_image.numpy()

def predict(image_path, model, top_k):
    """ Use the model to predict the top N classes
        for the image specified by image_path """
    
    original_image = Image.open(image_path)
    original_image_array = np.asarray(original_image)    
    processed_image = process_image(original_image_array)
    probs = model.predict(np.expand_dims(processed_image, axis=0))    
    top = tf.math.top_k(probs, top_k)
    return top.values.numpy()[0], top.indices.numpy()[0]
    
def get_flower_names(labels, label_map_path):
    """ Return the flower names, given the label/name mapping """
    
    class_names = []
    
    with open(label_map_path) as json_file:
        class_names = json.load(json_file)
    
    flower_names = [class_names[str(x+1)] for x in labels]
        
    return flower_names

def load_model(path_to_model):
    """ Load the model"""
    
    return tf.keras.models.load_model(path_to_model, custom_objects={'KerasLayer':hub.KerasLayer})