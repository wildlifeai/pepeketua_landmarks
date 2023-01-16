import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import cv2
import time
import copy
import sklearn.model_selection as skm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utils
from RotationGenerator import RotationGenerator
import imgaug as ia
import imgaug.augmenters as iaa

DB_FILE_PATH = r"tests\image_path_anotations_db.pkl"
IMAGE_SIZE = (256, 256)
IMAGE_SIZE_WIDTH_INDEX = 0
IMAGE_SIZE_HEIGHT_INDEX = 1
DIR_PATH = r"D:\Archeys_frogs\whareorino_a\Grid A\Individual Frogs"


def show_image(img):
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    df = pd.read_pickle(DB_FILE_PATH)
    df.image_path = df.image_path.apply(
        lambda path: "D:\\" + os.sep.join(os.path.normpath(path).split(os.sep)[5:])
    )

    gt = RotationGenerator(
        dataframe=df,
        x_col="image_path",
        y_col=df.columns.to_list()[1:],
        color_mode="rgb",
        target_size=IMAGE_SIZE,
        batch_size=10,
        training=True,
        rotate_90=[0, 1, 2, 3],
        resize_points=True,
        normalize_rotation=True,
    )

    # x = gt.__getitem__(2)
    # _, theta = utils.cart2pol(y[0][0], y[0][1])
    # theta = utils.positive_deg_theta(theta)
    # img = iaa.Affine(rotate = theta)(images = x[0:1])[0]

    for j in range(5):
        x, y = gt.__getitem__(2)
        for i in range(1):
            rho, theta = utils.cart2pol(y[i][0], y[i][1])
            theta = utils.positive_deg_theta(theta)
            print("The rotation is: {0}, {1}".format(theta, rho))
            show_image(x[i])


if __name__ == "__main__":
    main()
