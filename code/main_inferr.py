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
from LocalizationPointAccuracy import LocalizationPointAccuracy
from CNNBlock import CNNBlock

BATCH_SIZE = 3
IMAGE_SIZE = (224, 224)
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
    image_path = os.path.join('landmark_images','new_image_{0}.jpg').format(nn)
    cv2.imwrite(image_path, img)
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
    model.load_weights(model_path)
    return model


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
        default = "/info/prediction.pkl")
    parser.add_argument('-s', '--save_images', 
        action = 'store_true',
        help="If to save images with landmark detection, will save to landmark_images",
        default = False)
    return parser.parse_args()


def main(dir_path, output_path, save_images):
    model_path = r"model/model_weights_landmark_714_check"
    rot_model_path = r"model/model_weights_rot_10"
    if not os.path.exists('landmark_images'):
        os.makedirs('landmark_images')

    print("Reading Dir")
    image_paths = get_image_path(dir_path)

    if len(image_paths) == 0:
        raise argparse.ArgumentTypeError("{0} has no images".format(dir_path))

    image_df = get_image_df(image_paths)
    gpred_rot = RotationGenerator(dataframe = image_df,
                        x_col = "image_path",
                        y_col = image_df.columns.to_list()[1:],
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

    image_df['rotation'] = - rot_prediction

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


    print("Loading Model")
    model = load_model(model_path)

    print("Predicting")
    prediction = model.predict(gpred)

    prediction = fix_prediction_order(prediction)
    # Test predictions
    pred_df = pd.concat([image_df.image_path, pd.DataFrame(prediction), image_df.width_size, image_df.height_size, image_df.rotation], axis = 1)

    labels_column_names = pred_df.columns.to_list()[1:-2]

    pred_original_size = gpred.create_final_labels(pred_df[pred_df.columns.to_list()[1:]])

    pred_df[pred_df.columns.to_list()[1:-3]] = pred_original_size
    
    labels_column_names = pred_df.columns.to_list()[1:-3]

    
    if save_images:
        # Save predictions on original size
        for j, image_path in enumerate(pred_df.image_path):
            im = cv2.imread(image_path)
            labels = pred_df.iloc[j][labels_column_names].to_list()
            thickness = math.ceil(pred_df.iloc[j].height_size / 224) + 1
            _ = show_labels(im, labels, radius = 1, thickness = thickness)


    change_column_name_dict = {i : config.COLS_DF_NAMES[i] for i in range(0, len(config.COLS_DF_NAMES))}
    print("Creating output pickle")
    pred_df.rename(columns = change_column_name_dict, inplace = True)
    pred_df.to_pickle(output_path)


if __name__ == '__main__':
    args = create_arg_parse()
    main(args.dir_path, args.output_path, args.save_images)