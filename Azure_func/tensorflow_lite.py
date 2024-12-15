#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input



model_path = "../models/xception_v1_46_0.874.keras"


model = keras.models.load_model(model_path)


img = load_img(
    "week8/test/pants/1b5f2882-e33e-4efc-b469-bcf87b9f53ed.jpg",
    target_size=(299, 299)
)


img


X = np.array([img])

X = preprocess_input(X)



pred = model.predict(X)



classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants' ,
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt']


dict(zip(classes, pred[0]))


# ### Convert model to Tensorflow Lite

import tensorflow.lite as tflite


converter = tflite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('../models/clothing-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


interpreter = tflite.Interpreter(model_path='../models/clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)



preds


# ### Removing Tensorflow dependencies

from PIL import Image


img_path = "../week8/test/pants/1b5f2882-e33e-4efc-b469-bcf87b9f53ed.jpg"


with Image.open(img_path) as img:
    img = img.resize((299, 299), Image.NEAREST)


# Get the code for preprocessing images from keras github repo
def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


X = np.array([img], dtype='float32')

X = preprocess_input(X)

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

dict(zip(classes, preds[0]))


# ### Alternative way of doing this

get_ipython().system('pip install keras-image-helper')


pip install ai-edge-litert

from ai_edge_litert.interpreter import Interpreter
from keras_image_helper import create_preprocessor


interpreter = Interpreter(model_path='../models/clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


preprocessor = create_preprocessor('xception', target_size=(299, 299))


#img_path = "../week8/test/pants/1b5f2882-e33e-4efc-b469-bcf87b9f53ed.jpg"


def predict(url):
    X = preprocessor.from_path(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)


    return dict(zip(classes, preds[0]))