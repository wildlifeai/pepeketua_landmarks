import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

class LandMarkDataGenerator(keras.utils.Sequence):
    """ Generator for augmenting images and keeping track of 
        landmark points after augmentation
    """
    def __init__(self, dataframe,
                        x_col,
                        y_col,
                        color_mode = "rgb",
                        target_size = (256, 256),
                        batch_size = 32,
                        scale = (1.0,1.0),
                        translate_percent = (0,0),
                        rotate = (0,0),
                        shear = (0,0),
                        multiply = (1, 1),
                        multiply_per_channel = 0,
                        rescale = 1/255,
                        resize_points = False,
                        height_first = True,
                        training = False,
                        normalize_y = False,
                        preprocessing_function = None
                        ):
        """
        dataframe: dataframe holding the information 

        x_col: column name of image paths

        y_col: column name of label variables, should be x,y pairs
                if using resize_points last two column should be image size
                example: x_left_eye, y_left_eye, x_right_eye, y_right_eye, ..... , image_height, image_width

        color_mode: color mode of the image as used by ImageDateGenerator by keras

        target_size: Tuple of ints, indicating to what size images will be resized, default (256, 256)
                     where the size tuple is (height, width)

        batch_size: Int, Batch size to use, default 32

        rescale: float to multiply each element in the image, default 1/255

        resize_points: If to resize labels to new target size, used if points are not rescaled before hand, default False

        height_first: If using resize the df last two image size column struct is height,width, default True

        training: Determins if to return new label values or only images, used for training or evaluating model 
                  should be True for training, validating and test but should be False for prediction. 
                  default False

        normalize_y: Whether or not to devide point coordinates  by size of image
                     making point coordinates between 0-1, used when output layer's
                     activation is tanh etc.

        preprocessing_function: Preprocessing function to be past onto ImageDataGenerator
        
        Imgaug parameters for augmentation for more information
        visit imgaug documantations, all defualt values will
        lead to no augmentations, when using Tuples will chose a value
        in area uniformaly:

        scale: Scales image, default (1.0,1.0)
        translate_percent: Translates images by percentage, default (0,0)
        rotate: Rotates image, default (0,0)
        shear: Shears image, default (0,0)
        multiply: The value with which to multiply the pixel values in each image, default (1, 1)
        multiply_per_channel: default 0 ,Whether to use (imagewise) the same sample(s) for all channels
                                (False) or to sample value(s) for each channel (True). 
                                Setting this to True will therefore lead to different transformations per image and channel, 
                                otherwise only per image. If this value is a float p, then for p percent of all images per_channel will be treated as True. 
                                If it is a StochasticParameter it is expected to produce samples with values between 0.0 and 1.0, 
                                where values >0.5 will lead to per-channel behaviour (i.e. same as True).

        """
        # Augmentation parameters that will be given to imgaug
        # default values won't augment the image
        self.scale = scale
        self.translate_percent = translate_percent
        self.rotate = rotate
        self.shear = shear
        self.multiply = multiply
        self.multiply_per_channel = multiply_per_channel

        self.target_size = target_size
        self.batch_size = batch_size
        self.df = dataframe
        self.training = training
        self.resize_points = resize_points
        self.height_first = height_first
        self.normalize_y = normalize_y
        self.preprocessing_function = preprocessing_function

        # Image generator
        self.image_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = rescale,
        																preprocessing_function = preprocessing_function)
        # Vectorization function to create Keypoint objects for augmentation
        self._keypoint_vectorization = np.vectorize(lambda x,y: Keypoint(x = x, y = y))

        # Vectorization function to create Keypoint objects for augmentation with resizing
        # image_size is (height, width)
        self._keypoint_vectorization_resize = np.vectorize(lambda x, y, im_height, im_width: Keypoint(x = x, y = y).project((im_height, im_width), self.target_size))
        self._keypoint_vectorization_resize_back = np.vectorize(lambda x, y, im_height, im_width: Keypoint(x = x, y = y).project(self.target_size, (im_height, im_width)))
        self._keypoint_x_vectorization = np.vectorize(lambda x: (x.x / self.target_size[0]) if self.normalize_y else x.x)
        self._keypoint_y_vectorization = np.vectorize(lambda y: (y.y / self.target_size[1]) if self.normalize_y else y.y)
        # Vectorization function to check if points are inside the image (can get out if augmentation is too aggresive)
        self._keypoint_is_out_vectorization = np.vectorize(lambda p, x_max, y_max: p.x > x_max or p.y > y_max)



        # Creating image generator
        self.image_gen = self.image_datagen.flow_from_dataframe(dataframe = self.df,
                                                                x_col = x_col,
                                                                y_col = y_col,
                                                                color_mode = color_mode,
                                                                class_mode = "raw",
                                                                target_size = self.target_size,
                                                                batch_size = self.batch_size,
                                                                shuffle = self.training)
        self.on_epoch_end()


    def __getitem__(self, index):
        # Generate one batch of data
        images, labels = self.image_gen.next()

        if self.training:
            labels = self.create_training_labels(images, labels)
            if labels is None or len(labels) < self.batch_size:
                images, labels = self.__getitem__(index)
            return images, labels
        return images

    def create_final_labels(self, labels):
        # Resizes labels to original image size
        labels, image_size = self._fix_labels_get_image_size(labels)
        image_height, image_width = self._get_image_sizes_for_vectorizations(labels, image_size)
        kps = self._keypoint_vectorization_resize_back(labels[:, :, 0], labels[:, :, 1], image_height, image_width)
        points = np.array(kps)

        # Creating final x,y label pairs
        points_x = self._keypoint_x_vectorization(points[:])
        points_y = self._keypoint_y_vectorization(points[:])
        labels = np.reshape(np.dstack([points_x, points_y]), (labels.shape[0], labels.shape[1] * 2))

        return labels

    def _fix_labels_get_image_size(self, labels):
        labels = np.array(labels)
        labels = np.reshape(labels, (labels.shape[0], int(labels.shape[1] / 2), 2))
        image_size = labels[:,-1]
        labels = labels[:,:-1]
        return labels, image_size

    def _get_image_sizes_for_vectorizations(self, labels, image_size):
        height_index = 0 if self.height_first else 1
        width_index = 1 - height_index

        image_height = image_size[:, height_index]
        image_height = np.reshape(image_height, (image_height.shape[0], 1))
        image_width = image_size[:, width_index]
        image_width = np.reshape(image_width, (image_width.shape[0], 1))
        return image_height, image_width

    def create_training_labels(self, images, labels):
        # Creating all image point objects
        labels, image_size = self._fix_labels_get_image_size(labels)
        if self.resize_points:
            image_height, image_width = self._get_image_sizes_for_vectorizations(labels, image_size)
            kps = self._keypoint_vectorization_resize(labels[:, :, 0], labels[:, :, 1], image_height, image_width)
        else:
            kps = self._keypoint_vectorization(labels[:, :, 0], labels[:, :, 1])

        images, kps_aug = self.seq(images = images, keypoints = kps.tolist())
        
        points = np.array(kps_aug)

        # Validating that after augmentation all points are still in the image
        images_too_augmentated = self._keypoint_is_out_vectorization(points[:], self.target_size[0], self.target_size[1])
        images_too_augmentated = np.sum(images_too_augmentated, axis = 1)
        images = images[images_too_augmentated == 0]
        points = points[images_too_augmentated == 0]

        if len(points) == 0:
            return None

        # Creating final x,y label pairs
        points_x = self._keypoint_x_vectorization(points[:])
        points_y = self._keypoint_y_vectorization(points[:])
        labels = np.reshape(np.dstack([points_x, points_y]), (images.shape[0], labels.shape[1] * 2))

        return labels

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        # Runs on epoch ends
        # All the augmentations that will be applied
        self.seq = iaa.Sequential([
                    iaa.Multiply(self.multiply, per_channel = self.multiply_per_channel),
                    iaa.Affine(
                        scale = self.scale,
                        translate_percent = self.translate_percent,
                        rotate = self.rotate,
                        shear = self.shear
                        )
                    ])
        self.image_gen.on_epoch_end()






