import numpy as np
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from LandMarkDataGenerator import LandMarkDataGenerator
from LocalizationPointAccuracy import LocalizationPointAccuracy
import imgaug as ia
import imgaug.augmenters as iaa

DB_FILE_PATH = r"tests\image_path_anotations_db.pkl"
IMAGE_SIZE = (224, 224)
IMAGE_SIZE_WIDTH_INDEX = 0
IMAGE_SIZE_HEIGHT_INDEX = 1
DIR_PATH = r"D:\Archeys_frogs\whareorino_a\Grid A\Individual Frogs"

# plot diagnostic learning curves
def summarize_diagnostics(history, since_step = 10):
    gs = gridspec.GridSpec(3,3)
    fig = plt.figure(tight_layout = True)
    ax1 = fig.add_subplot(gs[0,:])
    # plot loss
    # plt.subplot(211)
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'][since_step:], color='blue', label='train')
    ax1.plot(history.history['val_loss'][since_step:], color='orange', label='test')
    # plot accuracy
    # plt.subplot(213)
    ax2 = fig.add_subplot(gs[1,:])
    ax2.set_title('Classification Accuracy')
    ax2.plot(history.history['angle_accuracy_radius_10'][since_step:], color='blue', label='train')
    ax2.plot(history.history['val_angle_accuracy_radius_10'][since_step:], color='orange', label='test')

    ax2 = fig.add_subplot(gs[2,:])
    ax2.set_title('Outside Radius Distance')
    ax2.plot(history.history['angle_outside_radius_0_distance'][since_step:], color='blue', label='train')
    ax2.plot(history.history['val_angle_outside_radius_0_distance'][since_step:], color='orange', label='test')

    plt.show()


def show_image(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class MyModel(keras.Model):
    """docstring for MyModel"""
    def __init__(self, **kwargs):
        super(MyModel, self).__init__()
        # self.backbone = tf.keras.applications.MobileNetV3Small(input_shape = (224, 224, 3), 
        #                                                         include_top = 'imagenet', 
        #                                                         weights = None, 
        #                                                         pooling = None)
        self.hidden = []
        num_filters = 512
        num_groups = 64
        st = 2
        counter = 0
        for i in range(6):
            if i > 0:
                self.hidden.append(tf.keras.layers.SeparableConv2D(num_filters, (3,3), activation = keras.activations.swish, strides = (st, st),
                                                                    activity_regularizer = tf.keras.regularizers.l2()))
            else:
                self.hidden.append(tf.keras.layers.SeparableConv2D(num_filters, (3,3), activation = keras.activations.swish, strides = (st, st)))
            self.hidden.append(tfa.layers.GroupNormalization(groups=int(num_groups), axis=3))
            if num_filters > 64 and counter % 2 != 0:
                num_filters /= 2
                num_groups /= 2
                st = 2
            else:
                st = 1
            counter += 1
            # if num_filters > 64:
            #     num_filters /= 2
            #     num_groups /= 2

            # if counter == 1:
            #     st = 1
            # counter += 1

        self.dense_1 = keras.layers.Dense(256, activation = 'relu')
        self.dense_2 = keras.layers.Dense(256, activation = 'relu')
        self.dense_3 = keras.layers.Dense(128, activation = 'relu')
        self.dense_4 = keras.layers.Dense(64, activation = 'relu')

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
        # x = self.dense_1(x)
        # x = self.drop_out_1(x)
        # x = self.dense_2(x)
        # x = self.drop_out_2(x)
        # x = self.dense_3(x)
        # x = self.drop_out_3(x)
        # x = self.dense_4(x)
        x = self.out(x)
        return x

def create_learning_rate():
    steps_in_epoch = 225
    base_lr = 0.00001
    boundaries = [steps_in_epoch * 50, steps_in_epoch * 100, steps_in_epoch * 150]
    values = [base_lr, base_lr * 0.1, base_lr * 0.001, base_lr * 0.0001]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    return learning_rate_fn

def create_my_model():
    model = MyModel()
    opt = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = opt, 
                  loss= tf.keras.losses.Huber(delta=10),
                  metrics = [LocalizationPointAccuracy(accuracy=True, radius=10),
                             LocalizationPointAccuracy(accuracy=True, radius=5)])
    return model

def load_diff_metrics():
    model = MyModel()
    opt = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = opt, 
                  loss= tf.keras.losses.Huber(delta=10),
                  metrics = [LocalizationPointAccuracy(accuracy=True, radius=8.75),
                             LocalizationPointAccuracy(accuracy=False, radius=8.75),
                             LocalizationPointAccuracy(accuracy=True, radius=4.375),
                             LocalizationPointAccuracy(accuracy=False, radius=4.375),
                             LocalizationPointAccuracy(accuracy=False, radius=0)])
    return model

def main():
    train_df = pd.read_pickle("tests/train_rot_db.pkl")
    val_df = pd.read_pickle("tests/val_rot_db.pkl")
    test_df = pd.read_pickle("tests/test_rot_db.pkl")

    train_df.image_path = train_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))
    val_df.image_path = val_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))
    test_df.image_path = test_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))


    gt = LandMarkDataGenerator(dataframe = train_df,
                        x_col = "image_path",
                        y_col = train_df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = IMAGE_SIZE,
                        batch_size = 5,
                        rotate = (-30,30),
                        multiply_per_channel = True,
                        training = True,
                        resize_points = True,
                        height_first = False,
                        specific_rotations = True)

    gv = LandMarkDataGenerator(dataframe = val_df,
                        x_col = "image_path",
                        y_col = val_df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = IMAGE_SIZE,
                        batch_size = 5,
                        rotate = (-20, 20),
                        training = True,
                        resize_points = True,
                        height_first = False,
                        specific_rotations = True)

    gtest = LandMarkDataGenerator(dataframe = test_df,
                        x_col = "image_path",
                        y_col = test_df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = IMAGE_SIZE,
                        batch_size = 5,
                        rotate = (-20, 20),
                        training = True,
                        resize_points = True,
                        height_first = False,
                        specific_rotations = True)


    my_model = create_my_model()
    weights_path = "weights/model_weights_{0}".format(np.random.randint(0,1000))
    # weights_path = "weights/model_weights_397"
    # my_model.load_weights(weights_path)
    check = keras.callbacks.ModelCheckpoint(weights_path, monitor="val_localization_point_accuracy_radius_8.75", mode = "max", save_weights_only=True, save_best_only=True)
    history = my_model.fit(gt, validation_data = gv, verbose = 1, epochs = 150, callbacks = [check])

    print(weights_path)

    my_model = load_diff_metrics()
    my_model.load_weights(weights_path)
    print("Evaluating on Train:")
    my_model.evaluate(gt)
    print("Evaluating on Val:")
    my_model.evaluate(gv)
    print("Evaluating on Test:")
    my_model.evaluate(gtest)

    summarize_diagnostics(history)

    # m = LocalizationPointAccuracy(accuracy=True, radius=30)
    # counter = 0
    # for j in range(math.ceil(len(val_df) / 32)):
    #     x,y_true = gv.__getitem__(1)
    #     y_pred = my_model.predict(x)
    #     for i in range(32):
    #         m.reset_state()
    #         m.update_state(y_true[i: i+1], y_pred[i: i+1])
    #         if m.result() < 1:
    #             counter += 1
    #             theta_true = tf.atan2(y_true[:, 1], y_true[: ,0])
    #             theta_pred = tf.atan2(y_pred[:, 1], y_pred[:, 0])
    #             diffs_1 = tf.abs(theta_true - theta_pred)
    #             diffs_2 = tf.abs(theta_true - theta_pred + (2 * np.pi))
    #             diffs_3 = tf.abs(theta_true - theta_pred - (2 * np.pi))
    #             diffs = tf.minimum(tf.minimum(diffs_1, diffs_2), diffs_3)
    #             diffs = diffs * 360 / (2 * np.pi)
    #             tt = theta_true * 360 / (2 * np.pi)
    #             tp = theta_pred * 360 / (2 * np.pi)
    #             print("True y: {0}".format(y_true[i]))
    #             print("Pred y: {0}".format(y_pred[i]))
    #             print("True theta: {0}".format(tt[i]))
    #             print("Pred theta: {0}".format(tp[i]))
    #             show_image(cv2.resize(x[i], (256, 256)))
    #             import pdb; pdb.set_trace()
    # print(counter)


    # count_val_m45_45 = 0
    # count_val_m45_m135 = 0
    # count_val_m135_135 = 0
    # count_val_45_135 = 0
    # for i in range(math.ceil(len(test_df) / 32)):
    #     x, y = gtest.__getitem__(3)
    #     for j in range(y.shape[0]):
    #         _, theta = utils.cart2pol(y[j][0], y[j][1])
    #         theta = np.rad2deg(theta)
    #         if -45 <= theta <= 45:
    #             count_val_m45_45 += 1
    #         elif 45 < theta <= 135:
    #             count_val_45_135 += 1
    #         elif 135 < theta <= 180 or -180 <= theta <= -135:
    #             count_val_m135_135 += 1
    #         elif -135 < theta < -45:
    #             count_val_m45_m135 += 1
    #         else:
    #             import pdb; pdb.set_trace()

    # total = count_val_m45_45 + count_val_m45_m135 + count_val_m135_135 + count_val_45_135
    # print("-45 to 45: {0}".format(count_val_m45_45))
    # print("45 to 135: {0}".format(count_val_45_135))
    # print("135 to -135: {0}".format(count_val_m135_135))
    # print("-45 to -135: {0}".format(count_val_m45_m135))
    # print("Total: {0}".format(total))
    # print("Len df: {0}".format(len(test_df)))



if __name__ == '__main__':
    main()
