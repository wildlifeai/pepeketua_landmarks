import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import keras
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import time
import copy
import math
import sklearn.model_selection as skm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
import utils
from LandMarkDataGenerator import LandMarkDataGenerator

DB_FILE_PATH = r"tests\image_path_anotations_db.pkl"
IMAGE_SIZE = (224, 224)
IMAGE_SIZE_WIDTH_INDEX = 0
IMAGE_SIZE_HEIGHT_INDEX = 1
DIR_PATH = r"D:\Archeys_frogs\whareorino_a\Grid A\Individual Frogs"


def show_labels(img, labels, labels_real = None, radius = 5, thickness = 1, radius_real = 10, color = (0, 0, 255), color_real = (0, 255, 0)):
    for i in range(0, len(labels), 2):
        point = np.round([labels[i], labels[i + 1]]).astype(int)
        point = tuple(point)
        img = cv2.circle(img, point, radius, color, thickness)
        if labels_real is not None:
            point = np.round([labels_real[i], labels_real[i + 1]]).astype(int)
            point = tuple(point)
            img = cv2.circle(img, point, radius, color_real, thickness)
            img = cv2.circle(img, point, radius_real, color_real, thickness)
    show_image(img)

def show_image(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    df = pd.read_pickle('prediction.pkl')

    df.image_path = df.image_path.apply(lambda path: 'D:\\error\\example_ims\\' + os.sep.join(os.path.normpath(path).split(os.sep)[2:]))

    gt = LandMarkDataGenerator(dataframe = df,
                        x_col = "image_path",
                        y_col = df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = IMAGE_SIZE,
                        batch_size = 10,
                        training = True,
                        resize_points = True,
                        height_first = False)


    # Test predictions on original size
    # for i in range(1):
    #     x, y = gt.__getitem__(i)
    #     for j in range(10):
    #         show_labels(x[j], y[j], labels_real = y[j], radius = 1, thickness = 1, radius_real=5)


    # Test predictions on original size
    for i, image_path in enumerate(df.image_path):
        im = cv2.imread(image_path)
        labels = np.array(df[config.COLS_DF_NAMES].iloc[i])
        _ = show_labels(im, labels, radius = 1, thickness = 1)



if __name__ == '__main__':
    main()
