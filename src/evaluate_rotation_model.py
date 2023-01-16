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


def load_rot_model(model_path):
    model = MyModelRotation()
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss=rotation_loss,
        metrics=[
            RotationAccuracy(accuracy=True, angle_radius=40),
            RotationAccuracy(accuracy=True, angle_radius=30),
            RotationAccuracy(accuracy=True, angle_radius=20),
            RotationAccuracy(accuracy=True, angle_radius=10),
            RotationAccuracy(accuracy=False, angle_radius=20),
            RotationAccuracy(accuracy=False, angle_radius=0),
        ],
    )
    model.load_weights(model_path)
    return model


def rotation_loss(y_true, y_pred, vec_norm=10):
    y_pred_norm = tf.norm(y_pred, axis=1, keepdims=1)
    point = tf.reduce_sum(tf.square(y_true - (y_pred / y_pred_norm)))
    norm = tf.reduce_sum(tf.abs(vec_norm - y_pred_norm))
    return 100 * point + norm


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
            self.hidden.append(
                keras.layers.Conv2D(
                    num_filters,
                    (3, 3),
                    activation=keras.activations.swish,
                    strides=(st, st),
                )
            )
            self.hidden.append(
                tfa.layers.GroupNormalization(groups=int(num_groups), axis=3)
            )
            if num_filters > 64:
                num_filters /= 2
                num_groups /= 2
            if counter == 1:
                st = 1
            counter += 1

        self.dense_1 = keras.layers.Dense(256, activation="relu")
        self.dense_2 = keras.layers.Dense(256, activation="relu")
        self.dense_3 = keras.layers.Dense(128, activation="relu")
        self.dense_4 = keras.layers.Dense(64, activation="relu")

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

    return np.roll(prediction, BATCH_SIZE, axis=0)


def main():
    rot_model_path = r"model/model_weights_rot_10"

    train_df = pd.read_pickle("tests/train_db.pkl")
    val_df = pd.read_pickle("tests/val_db.pkl")
    test_df = pd.read_pickle("tests/test_db.pkl")

    train_df.image_path = train_df.image_path.apply(
        lambda path: "D:\\" + os.sep.join(os.path.normpath(path).split(os.sep)[5:])
    )
    val_df.image_path = val_df.image_path.apply(
        lambda path: "D:\\" + os.sep.join(os.path.normpath(path).split(os.sep)[5:])
    )
    test_df.image_path = test_df.image_path.apply(
        lambda path: "D:\\" + os.sep.join(os.path.normpath(path).split(os.sep)[5:])
    )

    print("Loading Rotation Model")
    rot_model = load_rot_model(rot_model_path)

    validate_num = 10
    df_names = ["Val", "Test"]
    for j, df_to_test in enumerate([val_df, test_df]):
        gpred_rot = RotationGenerator(
            dataframe=df_to_test,
            x_col="image_path",
            y_col=df_to_test.columns.to_list()[1:],
            color_mode="rgb",
            target_size=ROT_IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            training=True,
            rotate_90=[0, 1, 2, 3],
            resize_points=True,
            height_first=False,
        )

        print("Evaluating on {0}:".format(df_names[j]))
        val_res = np.zeros((1, 7))
        for i in range(validate_num):
            val_res += rot_model.evaluate(gpred_rot)

        val_res /= validate_num
        print("*" * 60)
        print(
            (
                "loss: {0} - angle_accuracy_radius_40: {1} - angle_accuracy_radius_30: {2} - angle_accuracy_radius_20: {3} "
                + "- angle_accuracy_radius_10: {4} - angle_outside_radius_20_distance: {5} - angle_outside_radius_0_distance: {6}"
            ).format(*val_res[0])
        )
        print("*" * 60)


if __name__ == "__main__":
    main()
