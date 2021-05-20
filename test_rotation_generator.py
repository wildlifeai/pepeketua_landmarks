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
import sklearn.model_selection as skm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utils
from RotationGenerator import RotationGenerator

DB_FILE_PATH = r"tests\image_path_anotations_db.pkl"
IMAGE_SIZE = (256, 256)
IMAGE_SIZE_WIDTH_INDEX = 0
IMAGE_SIZE_HEIGHT_INDEX = 1
DIR_PATH = r"D:\Archeys_frogs\whareorino_a\Grid A\Individual Frogs"



def show_image(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    df = pd.read_pickle(DB_FILE_PATH)
    df.image_path = df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))

    gt = RotationGenerator(dataframe = df,
                        x_col = "image_path",
                        y_col = df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = IMAGE_SIZE,
                        batch_size = 10,
                        training = True,
                        resize_points = True,
                        normalize_rotation = True,
                        rotate_90 = (2,3))

    for j in range(2):
        x, y = gt.__getitem__(2)
        for i in range(5):
            rho, theta = utils.cart2pol(y[0][i], y[1][i])
            theta = utils.positive_deg_theta(theta)
            print("The rotation is: {0}".format(theta))
            show_image(x[i])



if __name__ == '__main__':
    main()
