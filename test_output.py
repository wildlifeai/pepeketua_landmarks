import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from RotationAccuracy import RotationAccuracy
from LocalizationPointAccuracy import LocalizationPointAccuracy
from CNNBlock import CNNBlock

BATCH_SIZE = 3
IMAGE_SIZE = (224, 224)
ROT_IMAGE_SIZE = (128, 128)


def load_model(model_path):
        model = MyModel()
        opt = keras.optimizers.Adam(learning_rate = 0.001)
        model.compile(optimizer = opt, 
                      loss= tf.keras.losses.Huber(delta=10),
                      metrics = [LocalizationPointAccuracy(accuracy=True, radius=8.75),
                                 LocalizationPointAccuracy(accuracy=False, radius=8.75),
                                 LocalizationPointAccuracy(accuracy=True, radius=4.375),
                                 LocalizationPointAccuracy(accuracy=False, radius=4.375),
                                 LocalizationPointAccuracy(accuracy=False, radius=0)])
        model.load_weights(model_path)
        return model

def load_rot_model(model_path):
    model = MyModelRotation()
    opt = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = opt, 
              loss= rotation_loss,
              metrics = [RotationAccuracy(accuracy=True, angle_radius=40),
                         RotationAccuracy(accuracy=True, angle_radius=30),
                         RotationAccuracy(accuracy=True, angle_radius=20),
                         RotationAccuracy(accuracy=True, angle_radius=10),
                         RotationAccuracy(accuracy=False, angle_radius=20),
                         RotationAccuracy(accuracy=False, angle_radius=0)])
    model.load_weights(model_path)
    return model

def rotation_loss(y_true, y_pred, vec_norm = 10):
    y_pred_norm = tf.norm(y_pred, axis = 1, keepdims = 1)
    point = tf.reduce_sum(tf.square(y_true - (y_pred / y_pred_norm)))
    norm = tf.reduce_sum(tf.abs(vec_norm - y_pred_norm))
    return 100 * point + norm

class MyModel(keras.Model):
    """docstring for MyModel"""
    def __init__(self, **kwargs):
        super(MyModel, self).__init__()
        # self.backbone = tf.keras.applications.MobileNetV3Small(input_shape = (224, 224, 3), 
        #                                                         include_top = False, 
        #                                                         weights = 'imagenet', 
        #                                                         pooling = None)

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
        self.dense_3 = keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())
        self.dense_4 = keras.layers.Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2())

        self.drop_out_1 = keras.layers.Dropout(0.55)
        self.drop_out_2 = keras.layers.Dropout(0.55)
        self.drop_out_3 = keras.layers.Dropout(0.55)
        self.out = keras.layers.Dense(12)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = inputs
        # x = self.avg_layer(x)
        for layer in self.hidden:
          x = layer(x)
        # x = self.backbone(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        # x = self.drop_out_1(x)
        x = self.dense_2(x)
        # x = self.drop_out_2(x)
        # x = self.dense_3(x)
        # x = self.drop_out_3(x)
        # x = self.dense_4(x)
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


def main():
        model_path = r"model/model_weights_landmark_714_check"
        rot_model_path = r"model/model_weights_rot_10"

        train_df = pd.read_pickle("tests/train_db.pkl")
        val_df = pd.read_pickle("tests/val_db.pkl")
        test_df = pd.read_pickle("tests/test_db.pkl")

        train_df.image_path = train_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))
        val_df.image_path = val_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))
        test_df.image_path = test_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))

        print("Loading Rotation Model")
        rot_model = load_rot_model(rot_model_path)
        print("Loading Model")
        model = load_model(model_path)

        for df_to_test in [val_df]:
            gpred_rot = RotationGenerator(dataframe = df_to_test,
                            x_col = "image_path",
                            y_col = df_to_test.columns.to_list()[1:],
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

            df_to_test['rotation'] = - rot_prediction

            gpred = LandMarkDataGenerator(dataframe = df_to_test,
                            x_col = "image_path",
                            y_col = df_to_test.columns.to_list()[1:],
                            color_mode = "rgb",
                            target_size = IMAGE_SIZE,
                            batch_size = BATCH_SIZE,
                            training = True,
                            resize_points = True,
                            height_first = False,
                            specific_rotations = True)

            m = LocalizationPointAccuracy(radius = 8)
            for i in range(gpred.__len__()):
                x,y = gpred.__getitem__(1)
                pred = model.predict(x)
                for j in range(BATCH_SIZE):
                    m.reset_states()
                    m.update_state(y[j], pred[j])
                    if m.result() < 0.99:
                        utils.show_labels(x[j], pred[j], labels_real = y[j], radius = 1, thickness = 1, radius_real=8, show_img = True)

if __name__ == '__main__':
    main()
