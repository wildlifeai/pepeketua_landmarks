import math
import os
import argparse
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import pandas as pd
import numpy as np

import config
import utils
from LandMarkDataGenerator import LandMarkDataGenerator
from RotationGenerator import RotationGenerator

BATCH_SIZE = 3
IMAGE_SIZE = (224, 224)
ROT_IMAGE_SIZE = (128, 128)

class MyModel(keras.Model):
    """docstring for MyModel"""
    def __init__(self, **kwargs):
        super(MyModel, self).__init__()
        self.hidden = []
        num_filters = 64
        num_groups = 16
        st = 2

        for i in range(8):
            self.hidden.append(tf.keras.layers.SeparableConv2D(num_filters, (3,3), activation = keras.activations.swish, strides = (st, st)))
            self.hidden.append(tfa.layers.GroupNormalization(groups=int(num_groups), axis=3))
            if num_filters <= 512 and i % 2 != 0:
                num_filters *= 2
                num_groups *= 2
                st = 2
            else:
                st = 1


        self.dense_1 = keras.layers.Dense(256, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())
        self.dense_2 = keras.layers.Dense(256, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())

        self.out = keras.layers.Dense(12)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
          x = layer(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.out(x)
        return x

class MyModelRotation(keras.Model):
    """docstring for MyModelRotation"""
    def __init__(self, **kwargs):
        super(MyModelRotation, self).__init__()
        self.hidden = []
        num_filters = 256
        num_groups = 64
        st = 2
        counter = 0
        for i in range(6):
            self.hidden.append(keras.layers.Conv2D(num_filters, (3,3), activation = keras.activations.swish, strides = (st, st)))
            self.hidden.append(tfa.layers.GroupNormalization(groups=int(num_groups), axis=3))
            if num_filters > 64:
                num_filters /= 2
                num_groups /= 2
            if counter == 1:
                st = 1
            counter += 1

        self.dense_1 = keras.layers.Dense(256, activation = 'relu')
        self.dense_2 = keras.layers.Dense(256, activation = 'relu')
        self.dense_3 = keras.layers.Dense(128, activation = 'relu')
        self.dense_4 = keras.layers.Dense(64, activation = 'relu')

        self.drop_out_1 = keras.layers.Dropout(0.25)
        self.drop_out_2 = keras.layers.Dropout(0.25)

        self.out = keras.layers.Dense(2)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
          x = layer(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.drop_out_1(x)
        x = self.dense_2(x)
        x = self.drop_out_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.out(x)
        return x

def fix_prediction_order(prediction):
    if len(prediction) <= BATCH_SIZE:
        return prediction

    return np.roll(prediction, BATCH_SIZE, axis = 0)

def get_image_df(image_paths):
    df = pd.DataFrame(image_paths, columns = ['image_path'])
    # Using loop and not apply because this is considerably faster
    image_sizes = []
    for im_path in df.image_path:
        im = cv2.imread(im_path)
        im_size = im.shape[:2]
        image_sizes.append(im_size)

    image_sizes = np.array(image_sizes)
    df['width_size'] = image_sizes[:, 1]
    df['height_size'] = image_sizes[:, 0]

    return df

def load_model(model_path):
    model = MyModel()
    model.load_weights(model_path)
    return model

def load_rot_model(model_path):
    model = MyModelRotation()
    model.load_weights(model_path)
    return model

def rotate(image_path, weigths_path = r"model/model_weights_rot_10"):
    image_df = get_image_df([image_path])
    print("Loading Rotation Model")
    rot_model = load_rot_model(weigths_path)
    gpred_rot = RotationGenerator(dataframe = image_df,
                        x_col = "image_path",
                        y_col = image_df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = ROT_IMAGE_SIZE,
                        batch_size = BATCH_SIZE,
                        training = False,
                        resize_points = True,
                        height_first = False)
    print("Predicting rotation")
    rot_prediction = rot_model.predict(gpred_rot)

    rot_prediction = fix_prediction_order(rot_prediction)
    pred_theta = np.arctan2(rot_prediction[:, 1], rot_prediction[:, 0])
    rot_prediction = utils.positive_deg_theta(pred_theta)

    image_df['rotation'] = - rot_prediction

    return image_df, rot_prediction

def find_landmarks(image_df, weigths_path = r"model/model_weights_landmark_714_check"):
    gpred = LandMarkDataGenerator(dataframe = image_df,
                        x_col = "image_path",
                        y_col = image_df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = IMAGE_SIZE,
                        batch_size = BATCH_SIZE,
                        training = False,
                        resize_points = True,
                        height_first = False,
                        specific_rotations = True)
    print("Loading Landmark Model")
    model = load_model(weigths_path)
    print("Predicting Landmarks")
    prediction = model.predict(gpred)

    prediction = fix_prediction_order(prediction)

    pred_df = pd.concat([image_df.image_path, pd.DataFrame(prediction), image_df.width_size, image_df.height_size, image_df.rotation], axis = 1)

    labels_column_names = pred_df.columns.to_list()[1:-2]

    pred_original_size = gpred.create_final_labels(pred_df[pred_df.columns.to_list()[1:]])

    pred_df[pred_df.columns.to_list()[1:-3]] = pred_original_size

    return pred_df

def show_labels(img, labels, radius = 5, thickness = 5, color = (0, 0, 255)):
    labels = labels.iloc[0][labels.columns.to_list()[1:-3]].to_list()
    for i in range(0, len(labels), 2):
        point = np.round([labels[i], labels[i+1]]).astype(int)
        point = tuple(point)
        img = cv2.circle(img, point, radius, color, thickness)

    return img
