## import os
import time
import sys

import math
import numpy as np
from keras import backend as K

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard



number_of_classes = 10
train_net = 'vgg16'
#args.lr=0.0001
lr=0.0001

def tanh_loss(y_true, y_pred):
    loss = -0.5*((1-y_true)*K.log(1-y_pred) + (1+y_true)*K.log(1+y_pred))
    return K.mean(loss,axis=-1)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = Flatten()(base_model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
cls_preds = Dense(number_of_classes, activation = 'softmax',name = 'class_prediction')(x)


orientation1_preds =  Dense(1, activation = 'tanh', name = 'orientation1')(x)
orientation2_preds =  Dense(1, activation = 'tanh', name = 'orientation2')(x)
orientation3_preds =  Dense(1, activation = 'tanh', name = 'orientation3')(x)

model = Model(input=base_model.input, output=[cls_preds, orientation1_preds, orientation2_preds, orientation3_preds],
              name = "%s"%(train_net))
model.summary()
print("Model name: %s"%(model.name))
optimizer = SGD(lr=lr, decay=1e-4, nesterov=True)
losses=['categorical_crossentropy', tanh_loss, tanh_loss, tanh_loss]
lossWeights = {'class_prediction':4.0,'orientation1':1.0,'orientation2':1.0,'orientation3':1.0}
model.compile(optimizer=optimizer, loss = losses, loss_weights = lossWeights, metrics=["accuracy"])
                 
