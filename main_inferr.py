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

from LandMarkDataGenerator import LandMarkDataGenerator
from LocalizationPointAccuracy import LocalizationPointAccuracy
from CNNBlock import CNNBlock

BATCH_SIZE = 3
IMAGE_SIZE = (256, 256)
COLS_DF_NAMES = ['x_Left_eye', 'y_Left_eye', 'x_Left_front_leg', 'y_Left_front_leg', 
        'x_Right_eye', 'y_Right_eye', 'x_Right_front_leg', 'y_Right_front_leg', 
        'x_Tip_of_snout', 'y_Tip_of_snout', 'x_Vent', 'y_Vent']

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
	return img


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
	model.load_weights("./model/my_model_7_weights")
	return model


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


def main(dir_path, output_path):
	model_path = r"model/my_model_7_weights"

	print("Reading Dir")
	image_paths = get_image_path(dir_path)

	if len(image_paths) == 0:
		raise argparse.ArgumentTypeError("{0} has no images".format(dir_path))

	image_df = get_image_df(image_paths)
	gpred = LandMarkDataGenerator(dataframe = image_df,
                        x_col = "image_path",
                        y_col = image_df.columns.to_list()[1:],
                        color_mode = "rgb",
                        target_size = IMAGE_SIZE,
                        batch_size = BATCH_SIZE,
                        training = False,
                        resize_points = True,
                        height_first = False)


	print("Loading Model")
	model = load_model(model_path)

	print("Predicting")
	prediction = model.predict(gpred)

	prediction = fix_prediction_order(prediction)

	# Test predictions
	pred_df = pd.concat([image_df.image_path, pd.DataFrame(prediction), image_df.width_size, image_df.height_size], axis = 1)

	labels_column_names = pred_df.columns.to_list()[1:-2]

	ordered_predictions = []

	pred_original_size = gpred.create_final_labels(pred_df[pred_df.columns.to_list()[1:]])

	pred_df[pred_df.columns.to_list()[1:-2]] = pred_original_size
	
	labels_column_names = pred_df.columns.to_list()[1:]

	# Test predictions on original size
	# for i, image_path in enumerate(pred_df.image_path):
	# 	im = cv2.imread(image_path)
	# 	labels = pred_df.iloc[i][labels_column_names].to_list()
	# 	_ = show_labels(im, labels, radius = 1, thickness = 10)


	change_column_name_dict = {i : COLS_DF_NAMES[i] for i in range(0, len(COLS_DF_NAMES))}

	print("Creating output pickle")
	pred_df.rename(columns = change_column_name_dict, inplace = True)
	pred_df.to_pickle(output_path)


if __name__ == '__main__':
	args = create_arg_parse()
	main(args.dir_path, args.output_path)