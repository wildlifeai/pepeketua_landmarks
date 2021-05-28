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
IMAGE_SIZE = (256, 256)
ROT_IMAGE_SIZE = (128, 128)

def show_labels(img, labels, labels_real = None, radius = 5, thickness = 1, radius_real = 10, color = (0, 0, 255), color_real = (0, 255, 0)):
        for i in range(0, len(labels), 2):
                point = np.round([labels[i], labels[i+1]]).astype(int)
                point = tuple(point)
                img = cv2.circle(img, point, radius, color, thickness)
                if labels_real is not None:
                        point = np.round([labels_real[i], labels_real[i + 1]]).astype(int)
                        point = tuple(point)
                        img = cv2.circle(img, point, radius, color_real, thickness)
                        img = cv2.circle(img, point, radius_real, color_real, thickness)
          
        # Save the image 
        nn = np.random.randint(0,1000)
        cv2.imwrite('new_image_{0}.jpg'.format(nn), img)
        # show_image(img)
        return img

def show_image(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_path(dir_path):
        image_paths = []
        for (dir_path, dir_names, file_names) in os.walk(dir_path):
                for file_name in file_names:
                        file_path = os.sep.join([dir_path, file_name])
                        if is_image(file_path):
                                image_paths.append(file_path)

        return image_paths

def is_image(file_path):
        try:
                im = Image.open(file_path)
                im.close()
                return True
        except Exception as e:
                return False

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
        model = MyModel(12)
        opt = keras.optimizers.Adam(learning_rate = 0.0001)
        model.compile(optimizer = opt, 
                      loss= tf.keras.losses.Huber(delta=10),
                      metrics = [LocalizationPointAccuracy(accuracy=True, radius=10),
                                 LocalizationPointAccuracy(accuracy=False, radius=10),
                                 LocalizationPointAccuracy(accuracy=True, radius=5),
                                 LocalizationPointAccuracy(accuracy=False, radius=5),
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
    def __init__(self, num_coords, **kwargs):
        super(MyModel, self).__init__()
        self.cnn_block_1 = CNNBlock(1, 512)
        self.cnn_block_2 = CNNBlock(1, 512, strides = (2,2))
        self.cnn_block_3 = CNNBlock(1, 256)
        self.cnn_block_4 = CNNBlock(1, 256, strides= (2,2))
        self.cnn_block_5 = CNNBlock(1, 128)
        self.cnn_block_6 = CNNBlock(1, 128, strides= (2,2))
        self.out = keras.layers.Dense(num_coords)
        self.gn_one = tfa.layers.GroupNormalization(groups=4, axis=3)
        self.gn_two = tfa.layers.GroupNormalization(groups=4, axis=3)
        self.gn_three = tfa.layers.GroupNormalization(groups=4, axis=3)
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = self.cnn_block_1(inputs)
        x = self.cnn_block_2(x)
        x = self.gn_one(x)
        x = self.cnn_block_3(x)
        x = self.cnn_block_4(x)
        x = self.gn_two(x)
        x = self.cnn_block_5(x)
        x = self.cnn_block_6(x)
        x = self.gn_three(x)
        x = self.flatten(x)
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

def dir_exists(x):
        if not os.path.isdir(x):
                raise argparse.ArgumentTypeError("{0} does not exist".format(x))
        return x

def create_arg_parse():
        parser = argparse.ArgumentParser(description = "Predict landmarks for Archey's Frogs")
        parser.add_argument('-d', '--dir_path', required = True, 
                type = dir_exists, 
                help = "Directory with images for the model to predict")
        parser.add_argument('-o', '--output_path', 
                type=str, 
                help="File path to the output pickle file",
                default = "prediction.pkl")
        return parser.parse_args()


def main():
        model_path = r"model/my_model_7_weights"
        rot_model_path = r"model/model_weights_163"

        train_df = pd.read_pickle("tests/train_db.pkl")
        val_df = pd.read_pickle("tests/val_db.pkl")
        test_df = pd.read_pickle("tests/test_db.pkl")

        train_df.image_path = train_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))
        val_df.image_path = val_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))
        test_df.image_path = test_df.image_path.apply(lambda path: 'D:\\' + os.sep.join(os.path.normpath(path).split(os.sep)[5:]))

        df_to_test = test_df
        gpred_rot = RotationGenerator(dataframe = df_to_test,
                        x_col = "image_path",
                        y_col = df_to_test.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = ROT_IMAGE_SIZE,
                        batch_size = BATCH_SIZE,
                        training = False,
                        resize_points = True,
                        height_first = False)

        print("Loading Rotation Model")
        rot_model = load_rot_model(rot_model_path)
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


        print("Loading Model")
        model = load_model(model_path)
        model.evaluate(gpred)

if __name__ == '__main__':
    main()
