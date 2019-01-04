# Transfer Learning
# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

# TensorFlow JS (TODO)
# https://gogul09.github.io/software/mobile-net-tensorflow-js#predict-using-mobilenet-model

import pandas as pd
import numpy as np
import os
from tensorflow import keras

import matplotlib.pyplot as plt

import tensorflow as tf

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
#from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import glob

base_model = MobileNet(weights='imagenet', include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x) #dense layer 2
x = Dense(512, activation='relu')(x) #dense layer 3


# bad data, batch_size must = 1 or else we see ProgBar issue (not enough samples to fill a batch)
image_path = './images/train.bad'
batch_size = 1

classes = glob.glob(image_path + '/*')
num_classes = len(classes)

preds = Dense(num_classes, activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator = train_datagen.flow_from_directory(image_path, 
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=1)

saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./pipeline_tfserving")

loaded_model = tf.contrib.saved_model.load_keras_model(saved_model_path)

print('Classes: %s' % classes)

# Shark pic
img_path = './images/predict/shark.jpg' 

predict_img = image.load_img(img_path, target_size=(224, 224))
predict_img_array = image.img_to_array(predict_img)
predict_img_array = np.expand_dims(predict_img_array, axis=0)
predict_preprocess_img = preprocess_input(predict_img_array)
 
prediction = loaded_model.predict(predict_preprocess_img)
print('%s: %s' % (img_path, prediction[0]))
#print(classes[np.argmax(prediction[0])])

# This shows the following error:
#   ValueError: `decode_predictions` expects a batch of predictions (i.e. a 2D array of shape (samples, 1000)). Found array with shape: (1, 3)
# Possibly because the original model isn't saved/loaded with the tf.contrib.saved_model utility above (known issue)

# decoded_prediction = decode_predictions(prediction)
# print(decoded_prediction)

# Fregly pic
img_path = './images/predict/fregly.png'

predict_img = image.load_img(img_path, target_size=(224, 224))
predict_img_array = image.img_to_array(predict_img)
predict_img_array = np.expand_dims(predict_img_array, axis=0)
predict_preprocess_img = preprocess_input(predict_img_array)

prediction = loaded_model.predict(predict_preprocess_img)
print('%s: %s' % (img_path, prediction[0]))
#print(classes[np.argmax(prediction[0])])

# Cat pic
img_path = './images/predict/cat.jpg'

predict_img = image.load_img(img_path, target_size=(224, 224))
predict_img_array = image.img_to_array(predict_img)
predict_img_array = np.expand_dims(predict_img_array, axis=0)
predict_preprocess_img = preprocess_input(predict_img_array)

prediction = loaded_model.predict(predict_preprocess_img)
print('%s: %s' % (img_path, prediction[0]))
#print(classes[np.argmax(prediction[0])])
