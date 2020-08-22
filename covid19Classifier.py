from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Add, Activation, Multiply, concatenate
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support
import itertools
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras
import cv2
import numpy as np
import argparse
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

# path for train, test, validate
trainPath = 'dataset/train'
valPath = 'dataset/val'

# create datagenerator
trainDatagen = ImageDataGenerator(rescale=1./255)
testDatagen = ImageDataGenerator(rescale=1./255)
valDatagen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDatagen.flow_from_directory(trainPath,
                                                  target_size=(320, 320),
                                                  shuffle=True, seed=42, class_mode="categorical",
                                                  color_mode='rgb',
                                                  batch_size=16)

valGenerator = valDatagen.flow_from_directory(valPath,
                                              target_size=(320, 320),
                                              color_mode='rgb',
                                              batch_size=1, seed=42, class_mode="categorical",
                                              shuffle=False)

# check number of training set
trainDataDist = np.unique(trainGenerator.classes, return_counts=True)[1]
classMappingTrain = trainGenerator.class_indices
for key, val in classMappingTrain.items():
    print(
        f'The number of {key} images are {trainDataDist[val]} in training-set')
print('_____________________________________________________________________')

# check number of validation set
valDataDist = np.unique(valGenerator.classes, return_counts=True)[1]
classMappingVal = valGenerator.class_indices
for key, val in classMappingVal.items():
    print(
        f'The number of {key} images are {valDataDist[val]} in validation-set')


# train model with inceptionV3
basemodel = InceptionV3(weights='imagenet', include_top=False,
                        input_tensor=Input(shape=(320, 320, 3)))

# extend base model
headModel = basemodel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)  # pool_size=(4, 4)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# create model
model = Model(inputs=basemodel.input, outputs=headModel)
model.summary()
epochs = 5
lr = 1e-4
BS = 16
model.compile(loss="categorical_crossentropy", optimizer=Adam(
    lr=1e-4, decay=lr/epochs), metrics=["accuracy"])

# train model and save here
filepath = "./checkpoints/"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
callbacks_list = [checkpoint]
model.fit(trainGenerator,
          steps_per_epoch=320//BS,
          epochs=epochs,
          validation_data=valGenerator,
          validation_steps=26,
          callbacks=callbacks_list
          )

model.save('./models/')