import copy
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import sklearn.model_selection as skm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import time

import utils
from MultiGen import MultiGen
from RotationAccuracy import RotationAccuracy
from RotationGenerator import RotationGenerator

DB_FILE_PATH = r"tests\image_path_anotations_db.pkl"
IMAGE_SIZE = (128, 128)
IMAGE_SIZE_WIDTH_INDEX = 0
IMAGE_SIZE_HEIGHT_INDEX = 1
DIR_PATH = r"D:\Archeys_frogs\whareorino_a\Grid A\Individual Frogs"


# plot diagnostic learning curves
def summarize_diagnostics(history, since_step=10):
    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(gs[0, :])
    # plot loss
    # plt.subplot(211)
    ax1.set_title("Loss")
    ax1.plot(history.history["loss"][since_step:], color="blue", label="train")
    ax1.plot(history.history["val_loss"][since_step:], color="orange", label="test")
    # plot accuracy
    # plt.subplot(213)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("Classification Accuracy")
    ax2.plot(
        history.history["angle_accuracy_radius_10"][since_step:],
        color="blue",
        label="train",
    )
    ax2.plot(
        history.history["val_angle_accuracy_radius_10"][since_step:],
        color="orange",
        label="test",
    )

    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title("Outside Radius Distance")
    ax2.plot(
        history.history["angle_outside_radius_0_distance"][since_step:],
        color="blue",
        label="train",
    )
    ax2.plot(
        history.history["val_angle_outside_radius_0_distance"][since_step:],
        color="orange",
        label="test",
    )

    plt.show()


def show_image(img):
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class MyModel(keras.Model):
    """docstring for MyModel"""

    def __init__(self, **kwargs):
        super(MyModel, self).__init__()
        self.hidden = []
        num_filters = 256
        num_groups = 64
        st = 2
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
            if i == 1:
                st = 1

        self.avg_layer = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))

        self.dense_1 = keras.layers.Dense(256, activation="relu")
        self.dense_2 = keras.layers.Dense(256, activation="relu")
        self.dense_3 = keras.layers.Dense(128, activation="relu")
        self.dense_4 = keras.layers.Dense(64, activation="relu")

        self.drop_out_1 = keras.layers.Dropout(0.25)
        self.drop_out_2 = keras.layers.Dropout(0.25)
        self.drop_out_3 = keras.layers.Dropout(0.25)
        self.out = keras.layers.Dense(2)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = inputs
        # x = self.avg_layer(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.drop_out_1(x)
        x = self.dense_2(x)
        x = self.drop_out_2(x)
        x = self.dense_3(x)
        # x = self.drop_out_3(x)
        x = self.dense_4(x)
        x = self.out(x)
        return x


def rotation_loss(y_true, y_pred, vec_norm=10):
    y_pred_norm = tf.norm(y_pred, axis=1, keepdims=1)
    point = tf.reduce_sum(tf.square(y_true - (y_pred / y_pred_norm)))
    norm = tf.reduce_sum(tf.abs(vec_norm - y_pred_norm))
    return 100 * point + norm


def create_learning_rate():
    steps_in_epoch = 36
    base_lr = 0.0001
    boundaries = [steps_in_epoch * 50, steps_in_epoch * 100, steps_in_epoch * 150]
    values = [base_lr, base_lr * 0.1, base_lr * 0.001, base_lr * 0.0001]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )
    return learning_rate_fn


def create_my_model():
    model = MyModel()
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss=rotation_loss,
        metrics=[
            RotationAccuracy(accuracy=True, angle_radius=20),
            RotationAccuracy(accuracy=True, angle_radius=10),
            RotationAccuracy(accuracy=False, angle_radius=0),
        ],
    )
    return model


def load_diff_metrics():
    model = MyModel()
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
    return model


def main():
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

    gt = RotationGenerator(
        dataframe=train_df,
        x_col="image_path",
        y_col=train_df.columns.to_list()[1:],
        color_mode="rgb",
        target_size=IMAGE_SIZE,
        batch_size=32,
        rotate_90=[0, 1, 2, 3],
        rotate=(-20, 20),
        multiply_per_channel=True,
        training=True,
        resize_points=True,
        height_first=False,
    )

    gv = MultiGen(
        RotationGenerator,
        3,
        dataframe=val_df,
        x_col="image_path",
        y_col=val_df.columns.to_list()[1:],
        color_mode="rgb",
        target_size=IMAGE_SIZE,
        batch_size=32,
        rotate_90=[0, 1, 2, 3],
        training=True,
        resize_points=True,
        height_first=False,
    )

    gtest = RotationGenerator(
        dataframe=test_df,
        x_col="image_path",
        y_col=test_df.columns.to_list()[1:],
        color_mode="rgb",
        target_size=IMAGE_SIZE,
        batch_size=32,
        rotate_90=[0, 1, 2, 3],
        training=True,
        resize_points=True,
        height_first=False,
    )

    gval = RotationGenerator(
        dataframe=val_df,
        x_col="image_path",
        y_col=val_df.columns.to_list()[1:],
        color_mode="rgb",
        target_size=IMAGE_SIZE,
        batch_size=32,
        rotate_90=[0, 1, 2, 3],
        training=True,
        resize_points=True,
        height_first=False,
    )

    my_model = create_my_model()
    weights_path = "weights/model_weights_{0}".format(np.random.randint(0, 1000))
    weights_path = "weights/model_weights_rot_2"
    # my_model.load_weights(weights_path + '_10')
    check = keras.callbacks.ModelCheckpoint(
        weights_path + "_20",
        monitor="val_angle_accuracy_radius_20",
        mode="max",
        save_weights_only=True,
        save_best_only=True,
    )
    check10 = keras.callbacks.ModelCheckpoint(
        weights_path + "_10",
        monitor="val_angle_accuracy_radius_10",
        mode="max",
        save_weights_only=True,
        save_best_only=True,
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=70,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )
    history = my_model.fit(
        gt, validation_data=gv, verbose=1, epochs=200, callbacks=[check, check10, early]
    )

    # weights_path = "model/model_weights_rot"

    validate_num = 10
    my_model = load_diff_metrics()

    for weights in [weights_path + "_20", weights_path + "_10"]:
        my_model.load_weights(weights)
        print("Evaluating on Val:")
        val_res = np.zeros((1, 7))
        for i in range(validate_num):
            val_res += my_model.evaluate(gval)

        val_res /= validate_num
        print("*" * 60)
        print(weights)
        print(
            (
                "loss: {0} - angle_accuracy_radius_40: {1} - angle_accuracy_radius_30: {2} - angle_accuracy_radius_20: {3} "
                + "- angle_accuracy_radius_10: {4} - angle_outside_radius_20_distance: {5} - angle_outside_radius_0_distance: {6}"
            ).format(*val_res[0])
        )
        print("*" * 60)

    for weights in [weights_path + "_20", weights_path + "_10"]:
        my_model.load_weights(weights)
        print("Evaluating on Val:")
        val_res = np.zeros((1, 7))
        for i in range(validate_num):
            val_res += my_model.evaluate(gtest)

        val_res /= validate_num
        print("*" * 60)
        print(weights)
        print(
            (
                "loss: {0} - angle_accuracy_radius_40: {1} - angle_accuracy_radius_30: {2} - angle_accuracy_radius_20: {3} "
                + "- angle_accuracy_radius_10: {4} - angle_outside_radius_20_distance: {5} - angle_outside_radius_0_distance: {6}"
            ).format(*val_res[0])
        )
        print("*" * 60)

    # x, y = gv.__getitem__(0)
    # import pdb; pdb.set_trace()
    # print("Evaluating on Test:")
    # my_model.evaluate(gtest)

    # summarize_diagnostics(history)

    # m = RotationAccuracy(accuracy=True, angle_radius=30)
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


if __name__ == "__main__":
    main()
