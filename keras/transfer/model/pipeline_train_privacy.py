# Transfer Learning
# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

# TensorFlow JS (TODO)
# https://gogul09.github.io/software/mobile-net-tensorflow-js#predict-using-mobilenet-model

import pandas as pd
import numpy as np
import os
from tensorflow import keras

#import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
#from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import glob

from privacy.optimizers import dp_optimizer

print(tf.VERSION)
print(tf.keras.__version__)

base_model = MobileNet(weights='imagenet', include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x) #dense layer 2
x = Dense(512, activation='relu')(x) #dense layer 3


image_path = './images/train'
learning_rate = 0.08
batch_size = 8 
num_epochs = 5 
l2_norm_clip = 1.0
noise_multiplier = 1.12
microbatches = 2 # must evently divide batch_size

classes = sorted(glob.glob(image_path + '/*'))
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

dp_optimizer_class = dp_optimizer.make_optimizer_class(
    tf.train.AdamOptimizer
)

if batch_size % microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')


optimizer = dp_optimizer_class(
          learning_rate=learning_rate,
          noise_multiplier=noise_multiplier,
          l2_norm_clip=l2_norm_clip,
          num_microbatches=microbatches
)

#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=num_epochs)

saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./pipeline_tfserving")

loaded_model = tf.contrib.saved_model.load_keras_model(saved_model_path)

print('Classes: %s' % classes)

# Cat pic
img_path = './images/predict/cats/cat.jpg'

predict_img = image.load_img(img_path, target_size=(224, 224))
predict_img_array = image.img_to_array(predict_img)
predict_img_array = np.expand_dims(predict_img_array, axis=0)
predict_preprocess_img = preprocess_input(predict_img_array)

prediction = loaded_model.predict(predict_preprocess_img)
print('%s: %s' % (img_path, prediction[0]))
#print(classes[np.argmax(prediction[0])])

# Dog pic
img_path = './images/predict/dogs/dog.jpg'

predict_img = image.load_img(img_path, target_size=(224, 224))
predict_img_array = image.img_to_array(predict_img)
predict_img_array = np.expand_dims(predict_img_array, axis=0)
predict_preprocess_img = preprocess_input(predict_img_array)

prediction = loaded_model.predict(predict_preprocess_img)
print('%s: %s' % (img_path, prediction[0]))
#print(classes[np.argmax(prediction[0])])

# Horse pic
img_path = './images/predict/horses/horse.jpg'

predict_img = image.load_img(img_path, target_size=(224, 224))
predict_img_array = image.img_to_array(predict_img)
predict_img_array = np.expand_dims(predict_img_array, axis=0)
predict_preprocess_img = preprocess_input(predict_img_array)

prediction = loaded_model.predict(predict_preprocess_img)
print('%s: %s' % (img_path, prediction[0]))
#print(classes[np.argmax(prediction[0])])
