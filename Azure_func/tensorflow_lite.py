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


# In[23]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[24]:


preds


# ### Removing Tensorflow dependencies

# In[8]:


from PIL import Image


# In[11]:


img_path = "../week8/test/pants/1b5f2882-e33e-4efc-b469-bcf87b9f53ed.jpg"


# In[12]:


with Image.open(img_path) as img:
    img = img.resize((299, 299), Image.NEAREST)


# In[13]:


# Get the code for preprocessing images from keras github repo
def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# In[34]:


X = np.array([img], dtype='float32')

X = preprocess_input(X)


# In[35]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[36]:


dict(zip(classes, preds[0]))


# ### Alternative way of doing this

# In[38]:


get_ipython().system('pip install keras-image-helper')


# In[51]:


pip install ai-edge-litert


# In[53]:


from ai_edge_litert.interpreter import Interpreter
from keras_image_helper import create_preprocessor


# In[54]:


interpreter = Interpreter(model_path='../models/clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[41]:


preprocessor = create_preprocessor('xception', target_size=(299, 299))


# In[46]:


img_path = "../week8/test/pants/1b5f2882-e33e-4efc-b469-bcf87b9f53ed.jpg"


# In[47]:


X = preprocessor.from_path(img_path)


# In[48]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[49]:


dict(zip(classes, preds[0]))

